"""ARPO outcome normalization and entropy-adaptive branch decisions.

Reference: RUC-NLPIR/ARPO, e484f1d, ARPO/verl_arpo_entropy/verl/
trainer/ppo/core_algos.py and workers/rollout/vllm_rollout/
vllm_rollout_with_tools.py. Upstream uses Apache-2.0.
"""
from collections import defaultdict
import math

import torch


def branch_probability(current_entropy, initial_entropy, probability=0.5, weight=0.2):
    """Probability equivalent to upstream's clipped U - weight * entropy_delta.

    The explicit boundary cases retain the upstream clipping behavior: a base
    probability of one always branches, including when entropy has decreased.
    """
    if not 0 <= probability <= 1 or weight < 0:
        raise ValueError("Require 0 <= branch_probability <= 1 and entropy_weight >= 0")
    if probability == 1:
        return 1.0
    return max(0.0, min(1.0, probability + weight * (current_entropy - initial_entropy)))


def normalized_topk_entropy(logprobs, vocab_size, max_tokens=20):
    """Upstream proxy: sum top-k entropy over first 20 tokens / log(vocab).

    This is the upstream truncated entropy statistic, not full-vocabulary entropy
    or the sampled-token surprisal. vLLM also returns the sampled token if needed.
    """
    if vocab_size <= 1:
        raise ValueError("Entropy normalization requires vocab_size > 1")
    return sum(-math.exp(item.logprob) * item.logprob
               for position in (logprobs or [])[:max_tokens]
               for item in position.values()) / math.log(vocab_size)


@torch.no_grad()
def trajectory_advantages(group_ids, trajectory_ids, rewards, normalize=True, epsilon=1e-6):
    """Compute GRPO outcome advantages once per trajectory, then map to turns.

    Must run before batch padding/reordering. Shared branch prefixes are separate
    trajectory occurrences, matching upstream's full-trajectory training.
    """
    if not (len(group_ids) == len(trajectory_ids) == len(rewards)):
        raise ValueError("Trajectory fields must have equal lengths")
    groups = defaultdict(dict)
    for group, trajectory, reward in zip(group_ids, trajectory_ids, rewards):
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("Outcome rewards must be finite")
        previous = groups[group].setdefault(trajectory, reward)
        if previous != reward:
            raise ValueError("Inconsistent outcome reward within a trajectory")
    values = {}
    for group, trajectories in groups.items():
        scores = torch.tensor(list(trajectories.values()), dtype=torch.float32)
        mean = scores.mean() if len(scores) > 1 else scores.new_tensor(0.)
        std = scores.std() if len(scores) > 1 else scores.new_tensor(1.)
        adv = (scores - mean) / (std + epsilon) if normalize else scores - mean
        values.update({(group, trajectory): value.item()
                       for trajectory, value in zip(trajectories, adv)})
    return torch.tensor([values[group, trajectory]
                         for group, trajectory in zip(group_ids, trajectory_ids)])
