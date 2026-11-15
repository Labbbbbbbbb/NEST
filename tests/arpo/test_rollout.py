"""CPU integration tests: branching must fork state, not just change labels."""
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

from arpo.rollout import adaptive_multi_turn_loop
from verl import DataProto


class FakeEnvs:
    def __init__(self, size, max_steps):
        self.size, self.max_steps = size, max_steps
        self.envs = self
        self.active_masks = None
        self.forks = []
        self.executed = 0

    def reset(self, kwargs=None):
        self.states = [0] * self.size
        self.gamefile = ["same-game"] * self.size
        self.tasks = ["task"] * self.size
        self.pre_text_obs = ["obs-0"] * self.size
        self.memory = NS(_data=[[] for _ in range(self.size)])
        return self.observations(), [{} for _ in range(self.size)]

    def observations(self):
        return {"text": [f"obs-{x}" for x in self.states], "anchor": list(self.states), "image": None}

    def step(self, actions):
        rewards = np.zeros(self.size)
        for i in range(self.size):
            # Mirror the manager's per-slot memory; dormant histories are replaced
            # on activation, while only active simulator slots may execute actions.
            self.memory._data[i].append(actions[i])
            if self.active_masks[i]:
                self.executed += 1
                self.states[i] += 1
                rewards[i] = 10. if self.states[i] == self.max_steps else 0.
        self.pre_text_obs = self.observations()["text"]
        done = np.array([x >= self.max_steps for x in self.states])
        return self.observations(), rewards, done, [{"won": bool(x), "is_action_valid": True} for x in done]

    def fork(self, source, target):
        assert self.states[target] == 0
        assert 0 < self.states[source] < self.max_steps
        self.states[target] = self.states[source]
        self.forks.append((source, target, self.states[source]))

    def success_evaluator(self, **kwargs):
        return {"success_rate": np.array([items[-1]["won"] for items in kwargs["total_infos"]])}


class FakeCollector:
    def __init__(self, probability, max_steps=3):
        self.config = NS(algorithm=NS(arpo=NS(initial_rollouts=1, beam_size=2,
                                              branch_probability=probability, entropy_weight=.2),
                                      filter_groups=NS(enable=False)),
                         env=NS(rollout=NS(n=4), seed=0, max_steps=max_steps, env_name="alfworld/AlfredTWEnv"),
                         actor_rollout_ref=NS(rollout=NS(name="vllm")))
        self._arpo_rollout_number = 0
        self.tokenizer = NS(batch_decode=lambda responses, **kw: ["action"] * len(responses))

    def preprocess_batch(self, gen_batch, obs):
        size = len(gen_batch)
        return DataProto.from_dict(tensors={"input_ids": torch.ones(size, 2, dtype=torch.long),
                                           "attention_mask": torch.ones(size, 2, dtype=torch.long),
                                           "position_ids": torch.zeros(size, 2, dtype=torch.long)},
                                   non_tensors={"raw_prompt_ids": np.array([[1, 1]] * size, dtype=object)})


class FakeActor:
    world_size = 2

    def generate_sequences(self, inputs):
        size = len(inputs)
        assert inputs.meta_info["arpo_entropy"]
        assert size % self.world_size == 0
        return DataProto.from_dict(tensors={"responses": torch.tensor([[4, 2]] * size),
                                           "input_ids": torch.tensor([[1, 1, 4, 2]] * size),
                                           "attention_mask": torch.ones(size, 4, dtype=torch.long),
                                           "position_ids": torch.arange(4).repeat(size, 1),
                                           "prompts": inputs.batch["input_ids"]},
                                   non_tensors={"arpo_entropy": np.ones(size) * .1})


class ARPORolloutTests(unittest.TestCase):
    def run_rollout(self, probability, max_steps=3):
        collector = FakeCollector(probability, max_steps)
        env = FakeEnvs(4, max_steps)
        gen_batch = DataProto.from_dict(tensors={"input_ids": torch.ones(4, 2, dtype=torch.long)})
        result = adaptive_multi_turn_loop(collector, gen_batch, FakeActor(), env)
        trajectories, rewards, lengths, success, ids, calls = result
        self.assertEqual(len(set(ids)), 4)
        np.testing.assert_array_equal(rewards, np.full(4, 10.))
        np.testing.assert_array_equal(lengths, np.full(4, max_steps))
        np.testing.assert_array_equal(success["success_rate"], np.ones(4))
        for trajectory, identity in zip(trajectories, ids):
            self.assertEqual(len(trajectory), max_steps)
            self.assertTrue(all(row["traj_uid"] == identity for row in trajectory))
        return env, collector

    def test_branches_copy_simulator_state_and_complete_prefixes(self):
        env, collector = self.run_rollout(1.)
        self.assertEqual(len(env.forks), 3)
        self.assertEqual(collector._arpo_metrics["arpo/root_restarts"], 0)
        self.assertLess(env.executed, 12)

    def test_zero_branch_probability_restarts_unused_roots(self):
        env, collector = self.run_rollout(0.)
        self.assertFalse(env.forks)
        self.assertEqual(collector._arpo_metrics["arpo/root_restarts"], 3)
        self.assertEqual(env.executed, 12)

    def test_terminal_first_step_never_forks_a_finished_state(self):
        env, collector = self.run_rollout(1., max_steps=1)
        self.assertFalse(env.forks)
        self.assertEqual(env.executed, 4)


if __name__ == "__main__":
    unittest.main()
