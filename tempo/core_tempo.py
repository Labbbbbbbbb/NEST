"""Token prefix value advantages from fatebreaker/tempo (0a1c273).

Adapted from verl/trainer/ppo/core_algos.py::compute_tempo_outcome_advantage
(Apache-2.0). NEST stores one assistant turn per row; build one prefix trie per
task over complete trajectories before batch padding or token balancing.
"""
from collections import defaultdict
from dataclasses import dataclass, field
import math

import torch


@dataclass
class Prefix:
    children: dict = field(default_factory=dict)
    total: float = 0.0
    count: int = 0

    @property
    def mean(self):
        return self.total / self.count


@torch.no_grad()
def trajectory_prefix_advantages(responses, response_mask, group_ids, trajectory_ids,
                                 steps, rewards, normalize=True, epsilon=1e-6):
    """A[i,t] = (R[i] - group_mean + V(prefix+t) - V(prefix)) / group_std.

    Outcome and prefix statistics count each trajectory once, independent of turn
    count. Exact token prefixes persist across turns. A non-token turn delimiter
    prevents ambiguous concatenation of different action sequences. Upstream's
    singleton shortcut reduces to R[i] - group_mean once the parent is unique.
    No artificial branch sampling is needed: the trie uses observed prefixes.
    """
    nrows = len(responses)
    if responses.ndim != 2 or response_mask.shape != responses.shape:
        raise ValueError("responses and response_mask must have matching 2D shapes")
    if any(len(value) != nrows for value in (group_ids, trajectory_ids, steps, rewards)):
        raise ValueError("Trajectory fields must have equal lengths")
    groups = defaultdict(dict)
    tokens = responses.detach().cpu()
    masks = response_mask.detach().cpu().bool()
    for row, (group, trajectory, step, reward) in enumerate(zip(group_ids, trajectory_ids, steps, rewards)):
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("Outcome rewards must be finite")
        item = groups[group].setdefault(trajectory, {"reward": reward, "rows": {}})
        if item["reward"] != reward:
            raise ValueError("Inconsistent outcome reward within a trajectory")
        step = int(step)
        if step in item["rows"]:
            raise ValueError("Duplicate trajectory step: compute before batch padding")
        item["rows"][step] = row

    result = torch.zeros_like(responses, dtype=torch.float32)
    for trajectories in groups.values():
        scores = torch.tensor([item["reward"] for item in trajectories.values()])
        group_mean = scores.mean().item()
        scale = (scores.std().item() if len(scores) > 1 else 1.0) + epsilon if normalize else 1.0
        root = Prefix()
        paths = {}
        # Each node stores the mean outcome of the trajectories sharing it.
        for trajectory, item in trajectories.items():
            node = root
            node.total += item["reward"]
            node.count += 1
            path = []
            for step, row in sorted(item["rows"].items()):
                for column in masks[row].nonzero().flatten().tolist():
                    token = int(tokens[row, column])
                    child = node.children.setdefault(token, Prefix())
                    child.total += item["reward"]
                    child.count += 1
                    path.append((row, column, node, child))
                    node = child
                child = node.children.setdefault(("turn", step), Prefix())
                child.total += item["reward"]
                child.count += 1
                node = child
            paths[trajectory] = path
        for trajectory, path in paths.items():
            baseline = trajectories[trajectory]["reward"] - group_mean
            for row, column, parent, child in path:
                result[row, column] = (baseline + child.mean - parent.mean) / scale
    return result.to(responses.device)
