"""Entropy-adaptive ARPO rollout using fixed ALFWorld worker slots.

Forks occur after an environment/tool response, like the upstream tool rollout.
Each group starts with initial_rollouts and grows up to env.rollout.n leaves.
"""
from copy import deepcopy
import uuid

import numpy as np

from arpo.core_arpo import branch_probability
from agent_system.multi_turn_rollout.utils import to_list_of_dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


def adaptive_multi_turn_loop(collector, gen_batch, actor_rollout_wg, envs):
    config = collector.config
    options = config.algorithm.arpo
    group_size = int(config.env.rollout.n)
    initial = int(options.initial_rollouts)
    beam = int(options.beam_size)
    if not 1 <= initial <= group_size or beam < 2:
        raise ValueError("Require 1 <= initial_rollouts <= env.rollout.n and beam_size >= 2")
    if config.env.env_name != "alfworld/AlfredTWEnv":
        raise ValueError("ARPO state replay currently supports alfworld/AlfredTWEnv only")
    if config.algorithm.filter_groups.enable:
        raise ValueError("ARPO uses its own bounded sampler; disable filter_groups")
    if config.actor_rollout_ref.rollout.name != "vllm":
        raise ValueError("ARPO entropy collection requires the vLLM rollout backend")
    probability = float(options.branch_probability)
    weight = float(options.entropy_weight)
    branch_probability(0., 0., probability, weight)  # Validate before generation.
    batch_size = len(gen_batch)
    if batch_size % group_size:
        raise ValueError("ARPO expects contiguous, complete environment groups")
    rng = np.random.default_rng(int(config.env.seed) + collector._arpo_rollout_number)
    collector._arpo_rollout_number += 1
    obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop("env_kwargs", None))
    root_obs = deepcopy(obs)
    root_tasks = deepcopy(envs.tasks)
    root_text = deepcopy(envs.pre_text_obs)
    groups = [list(range(start, start + group_size)) for start in range(0, batch_size, group_size)]
    for slots in groups:
        if len({envs.gamefile[slot] for slot in slots}) != 1:
            raise RuntimeError("ARPO group workers reset to different games")
        if len({obs["text"][slot] for slot in slots}) != 1:
            raise RuntimeError("ARPO group workers have different initial observations")
    group_ids = np.empty(batch_size, dtype=object)
    trajectory_ids = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
    activated = np.zeros(batch_size, dtype=bool)
    active = np.zeros(batch_size, dtype=bool)
    for slots in groups:
        group_ids[slots] = str(uuid.uuid4())
        activated[slots[:initial]] = active[slots[:initial]] = True
    trajectories = [[] for _ in range(batch_size)]
    all_infos = [[] for _ in range(batch_size)]
    rewards_sum = np.zeros(batch_size, dtype=np.float32)
    lengths = np.zeros(batch_size, dtype=np.float32)
    calls = np.zeros(batch_size, dtype=np.float32)
    initial_entropy = np.full(batch_size, np.nan)
    branch_count = 0
    restart_count = 0
    generated_turns = 0
    entropy_sum = 0.0
    replayed_steps = 0

    # A group can use at most group_size * max_steps environment transitions.
    for _ in range(group_size * int(config.env.max_steps)):
        indices = np.flatnonzero(active)
        if len(indices) == 0:
            break
        selected_obs = {key: ([value[i] for i in indices] if value is not None else None)
                        for key, value in obs.items()}
        batch = collector.preprocess_batch(gen_batch.select_idxs(indices), selected_obs)
        non_tensor_keys = [key for key in ("raw_prompt_ids", "raw_prompt", "multi_modal_data", "tools_kwargs")
                           if key in batch.non_tensor_batch]
        inputs = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"],
                           non_tensor_batch_keys=non_tensor_keys)
        inputs.meta_info = dict(gen_batch.meta_info, arpo_entropy=True)
        padded, padding = pad_dataproto_to_divisor(inputs, actor_rollout_wg.world_size)
        output = unpad_dataproto(actor_rollout_wg.generate_sequences(padded), padding)
        batch = batch.union(output)
        if "arpo_entropy" not in batch.non_tensor_batch:
            raise RuntimeError("vLLM did not return ARPO top-k entropy; check worktree imports")
        entropies = np.asarray(batch.non_tensor_batch["arpo_entropy"], dtype=float)
        generated_turns += len(indices)
        entropy_sum += float(entropies.sum())
        missing = np.isnan(initial_entropy[indices])
        initial_entropy[indices[missing]] = entropies[missing]
        actions = ["look"] * batch_size
        decoded = collector.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        for index, action in zip(indices, decoded):
            actions[index] = action
        # Inactive workers stay at their exact previous state.
        envs.envs.active_masks = active.copy()
        try:
            next_obs, rewards, dones, infos = envs.step(actions)
        finally:
            envs.envs.active_masks = None
        rewards = np.asarray(rewards).reshape(-1)
        dones = np.asarray(dones).reshape(-1)
        batch.non_tensor_batch.update({
            "uid": group_ids[indices], "traj_uid": trajectory_ids[indices],
            "active_masks": np.ones(len(indices), dtype=object),
            "rewards": rewards[indices].astype(object),
            "is_action_valid": np.array([infos[i].get("is_action_valid", True) for i in indices]),
        })
        for local, row in enumerate(to_list_of_dict(batch)):
            index = indices[local]
            trajectories[index].append(row)
            all_infos[index].append(infos[index])
            rewards_sum[index] += rewards[index]
            lengths[index] += 1
            calls[index] += float(infos[index].get("tool_calling", 0))
            if dones[index] or lengths[index] >= config.env.max_steps:
                active[index] = False
        obs = next_obs
        entropy_by_slot = dict(zip(indices, entropies))
        for slots in groups:
            # Branch only from leaves which actually generated this round.
            sources = [slot for slot in slots if active[slot] and slot in entropy_by_slot]
            free = [slot for slot in slots if not activated[slot]]
            for source in sources:
                for _ in range(beam - 1):
                    if not free:
                        break
                    chance = branch_probability(entropy_by_slot[source], initial_entropy[source], probability, weight)
                    if rng.random() >= chance:
                        continue
                    target = free.pop(0)
                    envs.envs.fork(source, target)
                    replayed_steps += int(lengths[source])
                    # Copy the agent's history as well as the simulator state.
                    envs.memory._data[target] = deepcopy(envs.memory._data[source])
                    envs.pre_text_obs[target] = envs.pre_text_obs[source]
                    envs.tasks[target] = envs.tasks[source]
                    envs.gamefile[target] = envs.gamefile[source]
                    for key in obs:
                        if obs[key] is not None:
                            obs[key][target] = deepcopy(obs[key][source])
                    trajectories[target] = [dict(row, traj_uid=trajectory_ids[target])
                                            for row in trajectories[source]]
                    all_infos[target] = deepcopy(all_infos[source])
                    rewards_sum[target] = rewards_sum[source]
                    lengths[target] = lengths[source]
                    calls[target] = calls[source]
                    activated[target] = active[target] = True
                    branch_count += 1
            if not active[slots].any() and free:
                # Upstream restarts from the original prompt if no live leaf can
                # branch. Reserved workers have never stepped; restore only memory.
                target = free[0]
                activated[target] = active[target] = True
                envs.memory._data[target] = []
                envs.pre_text_obs[target] = root_text[target]
                envs.tasks[target] = root_tasks[target]
                for key in obs:
                    if obs[key] is not None:
                        obs[key][target] = deepcopy(root_obs[key][target])
                restart_count += 1
        if activated.all() and not active.any():
            break
    if not activated.all() or active.any():
        raise RuntimeError("ARPO exceeded its bounded rollout budget")
    success = envs.success_evaluator(total_infos=all_infos, total_batch_list=trajectories,
                                    episode_rewards=rewards_sum, episode_lengths=lengths)
    collector._arpo_metrics = {"arpo/branches": branch_count, "arpo/root_restarts": restart_count,
                               "arpo/trajectories": batch_size,
                               "arpo/generated_turns": generated_turns, "arpo/replayed_steps": replayed_steps,
                               "arpo/branch_entropy_mean": entropy_sum / max(generated_turns, 1)}
    return trajectories, rewards_sum, lengths, success, trajectory_ids, calls
