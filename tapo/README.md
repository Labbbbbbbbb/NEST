# NEST 的 TAPO 在线 ALFWorld 训练

本实现基于 [Sunrepe/TAPO](https://github.com/Sunrepe/TAPO)，核对版本为
`040a38e27f0ea4bfa161c76b92aaf24533e33a70`。主要对照：

- `openrlhf/datasets/utils.py`：任务组内的二元成功结果标准化。
- `openrlhf/trainer/tapo_trainer.py`：`TAPOLoss`、成功样本 CE，以及可选 KL。

## 运行

在 NEST 根目录提交（使用已有 NEST conda 环境、ALFWorld 数据和 CUDA 依赖）：

```bash
sbatch scripts/train_tapo.sh
```

脚本默认申请 project3-1 分区的 2 GPU、16 CPU、256G 主机内存。
CPU 和内存数值是启动配置，不是测量得到的最低需求；GPU 显存是否足够仍取决于卡型。
检查 `slurm-<jobid>.out` 中的初始化、验证和训练指标。

已有 GPU allocation 时，也可运行：

```bash
bash scripts/train_tapo.sh trainer.total_training_steps=1 trainer.val_before_train=False
```

这条命令用于完整训练链路的一步冒烟检查，会采集轨迹、反向传播并更新参数。
它仍然需要 ALFWorld 和模型就绪，单步也可能耗时数分钟。

使用本地缓存模型：

```bash
export MODEL_PATH=/absolute/path/to/Qwen2.5-1.5B-Instruct
sbatch --export=ALL scripts/train_tapo.sh
```

下载需要代理时，应确保计算节点可访问代理，再导出 `http_proxy` / `https_proxy`。
本入口不自动启动原 GiGPO 脚本中的 SSH 隧道。

只查看 Hydra 配置、不启动训练：

```bash
python3 -m verl.trainer.main_ppo --cfg job algorithm.adv_estimator=tapo \
  actor_rollout_ref.actor.policy_loss.loss_mode=tapo
```

## 算法

每个任务采样 K 条轨迹，使用环境的单轨迹成功标记 `success_rate`（0/1），
不把 ALFWorld 的 `10*won + goal_condition_success_rate` shaped reward 当作成功标签。
设轨迹结果为 s，组内均值为 μ、总体标准差为 σ：

```text
A = (s - μ) / σ             # σ > 1e-8
A = 0                      # 全成功、全失败或单轨迹组
M_t = response_mask_t * 1[H(πθ(·|context_t)) > entropy_threshold]
r_t = πθ(token_t|context_t) / (πref(token_t|context_t) + 1e-8)
L_TAPO = -mean_valid_rows(mean_selected_tokens(min(r_t*A, clip(r_t,1-ε,1+ε)*A)))
L = 0.8 * L_TAPO + 0.2 * L_CE_success
```

优势在完整采集批次上按 `uid` 分组、按 `traj_uid` 去重计算，然后再补齐、重排和分发。
同一轨迹的所有回答 token 共享优势。长度更长的轨迹不会在组均值中被重复计数。
全同结果组不会贡献 TAPO 或 CE；指标 `tapo/nonzero_advantage_fraction` 可观察有效信号占比。
整批都没有混合组时，会打印 `No mixed success/failure groups` 并跳过更新。
反复出现该信息时，可检查模型能力、任务难度和 GROUP_SIZE。

`πref` 是训练开始时加载、随后固定的参考模型；即使关闭 KL，也会创建并计算参考策略。
`old_log_probs` 仍由公共训练框架生成用于诊断，但 TAPO 损失的分母不会使用它。
熵是完整词表分布的熵，单位为 nats；0.01 是绝对阈值，使用严格 `>`，不是百分位数。
CE 仅作用于混合组中的成功回答，不做熵过滤。默认不添加 KL；需要时可开启
`actor_rollout_ref.actor.use_kl_loss=True` 并配置其系数。

主要参数可通过命令行覆盖：

```bash
sbatch scripts/train_tapo.sh \
  actor_rollout_ref.actor.policy_loss.tapo_entropy_threshold=0.1 \
  actor_rollout_ref.actor.policy_loss.tapo_epsilon=0.2 \
  actor_rollout_ref.actor.policy_loss.tapo_ce_coef=0.0 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
```

`tapo_ce_coef=0` 关闭成功样本 CE；如需单独的 TAPO 项，还可设 `tapo_loss_coef=1.0`。
默认每任务 8 条轨迹、8 个训练任务和 16 个验证任务。修改采样规模使用环境变量：

```bash
TRAIN_SIZE=8 VAL_SIZE=16 GROUP_SIZE=8 sbatch --export=ALL scripts/train_tapo.sh
```

环境 worker 共 `(TRAIN_SIZE * GROUP_SIZE + VAL_SIZE)` 个，各预留 0.1 Ray CPU；
还需给 TaskRunner 和 GPU worker 留出 CPU。增大规模时同时调整 Slurm CPU 申请和
`RAY_CPUS`，否则 Ray 可能等待资源。`N_GPUS` 必须与 Slurm GPU 申请一致。

## 从训练路径移除的 GiGPO 组件

- 不调用 `gigpo/core_gigpo.py`，GiGPO 改为仅在自身分支内导入。
- 不采集 TAPO 无需的 `anchor_obs`，不执行观察相似度分组。
- 不计算 step discounted returns，也不叠加 `step_advantage_w`。
- 不读取 `algorithm.gigpo.*` 来计算 TAPO 优势。
- 禁用无效动作奖励惩罚、reward KL 和基于 shaped reward 的动态组过滤。
- 无 Critic；使用 TAPO 优势和 TAPO token 损失替代 GiGPO 更新。

原 GiGPO 文件保留供原入口使用；新增 TAPO 入口不依赖这些专属计算。

## 与官方训练方式的差异及验证范围

官方训练器读取已采集的离线轨迹，按完整对话样本训练。这里复用 NEST 的在线
ALFWorld collector、vLLM rollout 和 FSDP actor，每轮重新采集，并将每个有效 turn
作为训练行。公式中的 reference ratio、clip 和熵筛选保持一致，但按 turn 平均会改变
整段轨迹内各 turn 的权重；梯度累积按 micro batch 的有效行均值计算。
公共 `adjust_batch` 还可能为并行整除复制少量行。因此这是在线适配版，不应宣称是
官方离线实验的逐样本等价复现，或把官方成绩视为本版本预期成绩。

验证命令：

```bash
python3 -m unittest discover -s tests/tapo -v
bash -n scripts/train_tapo.sh examples/tapo_trainer/run_alfworld.sh
```

已通过 7 项 CPU 测试、Python/Bash 语法检查和实际启动参数的 Hydra 配置解析。
另使用官方 `TAPOLoss` 类对照随机 logits，检查损失值、梯度和空筛选情况一致。
训练路径测试调用真实优势分发和 actor 更新方法，使用微型 CPU 策略代替 GPU 模型。
尚未执行完整的多 GPU + vLLM + ALFWorld 训练，需用上述单步命令进行集群验证。
