"""TAPO 的轨迹优势计算与 token 级策略损失。

改编自 Sunrepe/TAPO (040a38e) 的 datasets/utils.py 和 TAPOLoss。
参考策略是冻结的初始策略，而不是上一次 rollout 的策略。
NEST 将每个 assistant 回合表示为一行训练数据，因此一条轨迹可能对应多行。
"""
from collections import defaultdict

import torch


@torch.no_grad()
def trajectory_advantages(group_ids, trajectory_ids, successes):
    """按组对轨迹的二元成功结果做总体标准化，再映射回各训练行。

    group_ids、trajectory_ids、successes 是逐训练行对齐的等长序列。
    同组通常对应同一任务的多次采样；successes 必须是整条轨迹最终的
    成功标签（成功为 1，失败为 0），同一轨迹的所有回合必须使用同一标签。
    这是本实现沿用的二元结果约定，不是标准化公式本身只能处理二元数值。

    每条轨迹只参与一次组内统计，避免回合数多的轨迹权重更大。
    必须在收集到的步骤被填充、裁剪或分发前计算，以保留完整组内统计。
    与上游一致，全成功或全失败的组没有相对优劣，优势全部置零。
    返回逐行的一维 CPU 张量，同一轨迹的各行得到相同优势。
    """
    groups = defaultdict(dict)
    for group, trajectory, success in zip(group_ids, trajectory_ids, successes):
        score = float(success)
        # 拒绝连续奖励或成功率均值；这里需要每条轨迹实际的成功/失败结果。
        if score not in (0.0, 1.0):
            raise ValueError("TAPO requires binary per-trajectory success labels")
        previous = groups[group].get(trajectory, score)
        if previous != score:
            raise ValueError("Inconsistent success labels within a trajectory")
        groups[group][trajectory] = score
    normalized = {}
    for group, trajectories in groups.items():
        scores = torch.tensor(list(trajectories.values()), dtype=torch.float32)
        # A_i = (s_i - mean(s)) / std(s)，使用总体标准差（分母为 N）。
        std = scores.std(unbiased=False)
        values = (scores - scores.mean()) / std if std > 1e-8 else torch.zeros_like(scores)
        normalized.update({(group, trajectory): value.item()
                           for trajectory, value in zip(trajectories, values)})
    return torch.tensor([normalized[group, trajectory]
                         for group, trajectory in zip(group_ids, trajectory_ids)])


def sample_mean(values, mask):
    """先计算每行被选 token 的均值，再对有效行等权平均。

    values 和 mask 的形状为 [训练行数, token 数]；没有选中 token 的行
    不参与平均。所有行都为空时保留可微的零，保证 FSDP 反向传播可执行。
    """
    mask = mask.to(values.dtype)
    counts = mask.sum(-1)
    means = (values * mask).sum(-1) / counts.clamp_min(1)
    valid = (counts > 0).to(values.dtype)
    return (means * valid).sum() / valid.sum().clamp_min(1)


def compute_tapo_loss(log_prob, ref_log_prob, advantages, response_mask,
                      entropy, epsilon=0.2, entropy_threshold=0.01):
    """用熵筛选 token，并计算上游定义的裁剪策略损失及监控指标。

    log_prob/ref_log_prob 是已采样 token 在当前/冻结参考策略下的对数概率，
    与 response_mask、entropy 的形状均为 [训练行数, token 数]。
    advantages 应能广播到该形状，例如将逐行轨迹优势扩为 [训练行数, 1]。
    entropy 是各位置策略分布的熵，阈值采用绝对值，单位为 nat。
    参考概率和熵筛选均不传递梯度；熵不作为额外奖励加入损失。
    """
    if not 0 <= epsilon < 1 or entropy_threshold < 0:
        raise ValueError("Require 0 <= epsilon < 1 and entropy_threshold >= 0")
    log_prob = log_prob.float()
    ref_log_prob = ref_log_prob.detach().float()
    # 稳定计算 log(p_ref + 1e-8)，保留上游分母的数值平滑项。
    denominator = torch.logaddexp(ref_log_prob, ref_log_prob.new_tensor(1e-8).log())
    ratio = (log_prob - denominator).exp()
    # 只训练响应中熵严格大于阈值的位置。
    selected = response_mask.bool() & (entropy.detach() > entropy_threshold)
    # 取未裁剪/裁剪目标的较小者；最终取负号，将最大化目标转为最小化损失。
    objective = torch.minimum(ratio * advantages,
                              ratio.clamp(1 - epsilon, 1 + epsilon) * advantages)
    loss = -sample_mean(objective, selected)
    # selected_fraction 以全部响应 token 为分母，正负 token 数只统计选中位置。
    metrics = {
        "actor/tapo_loss": loss.detach().item(),
        "actor/tapo_selected_fraction": (selected.sum() / response_mask.sum().clamp_min(1)).item(),
        "actor/tapo_positive_tokens": (selected & (advantages > 0)).sum().item(),
        "actor/tapo_negative_tokens": (selected & (advantages < 0)).sum().item(),
    }
    return loss, metrics
