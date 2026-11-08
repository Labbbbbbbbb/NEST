"""Adapted from Sunrepe/TAPO (040a38e), datasets/utils.py and TAPOLoss.

The reference is the frozen initial policy, not the previous rollout policy.
NEST represents each assistant turn as a separate training row.
"""
from collections import defaultdict

import torch


@torch.no_grad()
def trajectory_advantages(group_ids, trajectory_ids, successes):
    """Population-standardized binary outcomes; count each trajectory once.

    Compute before padding, trimming or distributing the collected steps.
    Homogeneous groups contribute zero TAPO advantage, as in upstream.
    """
    groups = defaultdict(dict)
    for group, trajectory, success in zip(group_ids, trajectory_ids, successes):
        score = float(success)
        if score not in (0.0, 1.0):
            raise ValueError("TAPO requires binary per-trajectory success labels")
        previous = groups[group].get(trajectory, score)
        if previous != score:
            raise ValueError("Inconsistent success labels within a trajectory")
        groups[group][trajectory] = score
    normalized = {}
    for group, trajectories in groups.items():
        scores = torch.tensor(list(trajectories.values()), dtype=torch.float32)
        std = scores.std(unbiased=False)
        values = (scores - scores.mean()) / std if std > 1e-8 else torch.zeros_like(scores)
        normalized.update({(group, trajectory): value.item()
                           for trajectory, value in zip(trajectories, values)})
    return torch.tensor([normalized[group, trajectory]
                         for group, trajectory in zip(group_ids, trajectory_ids)])


def sample_mean(values, mask):
    """Mean of per-row token means, excluding rows with no selected tokens.

    Keep a differentiable zero when no rows contribute (safe for FSDP backward).
    """
    mask = mask.to(values.dtype)
    counts = mask.sum(-1)
    means = (values * mask).sum(-1) / counts.clamp_min(1)
    valid = (counts > 0).to(values.dtype)
    return (means * valid).sum() / valid.sum().clamp_min(1)


def compute_tapo_loss(log_prob, ref_log_prob, advantages, response_mask,
                      entropy, epsilon=0.2, entropy_threshold=0.01):
    """Upstream clipped token ratio and absolute entropy threshold (nats).

    Preserve upstream's p_ref + 1e-8 denominator using stable logaddexp.
    Entropy only controls selection; its gradient is not part of the loss.
    """
    if not 0 <= epsilon < 1 or entropy_threshold < 0:
        raise ValueError("Require 0 <= epsilon < 1 and entropy_threshold >= 0")
    log_prob = log_prob.float()
    ref_log_prob = ref_log_prob.detach().float()
    denominator = torch.logaddexp(ref_log_prob, ref_log_prob.new_tensor(1e-8).log())
    ratio = (log_prob - denominator).exp()
    selected = response_mask.bool() & (entropy.detach() > entropy_threshold)
    objective = torch.minimum(ratio * advantages,
                              ratio.clamp(1 - epsilon, 1 + epsilon) * advantages)
    loss = -sample_mean(objective, selected)
    metrics = {
        "actor/tapo_loss": loss.detach().item(),
        "actor/tapo_selected_fraction": (selected.sum() / response_mask.sum().clamp_min(1)).item(),
        "actor/tapo_positive_tokens": (selected & (advantages > 0)).sum().item(),
        "actor/tapo_negative_tokens": (selected & (advantages < 0)).sum().item(),
    }
    return loss, metrics
