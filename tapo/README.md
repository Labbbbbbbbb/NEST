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

脚本默认申请 project3-1 分区的 4 GPU、32 CPU、256G 主机内存。
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
export MODEL_PATH=/absolute/path/to/Qwen2.5-7B-Instruct
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
L = 0.8 * L_TAPO + 0.2 * L_CE_success + 0.01 * L_KL
```

优势在完整采集批次上按 `uid` 分组、按 `traj_uid` 去重计算，然后再补齐、重排和分发。
同一轨迹的所有回答 token 共享优势。长度更长的轨迹不会在组均值中被重复计数。
全同结果组不会贡献 TAPO 或 CE；指标 `tapo/nonzero_advantage_fraction` 可观察有效信号占比。
整批都没有混合组时，会打印 `No mixed success/failure groups` 并跳过更新。
反复出现该信息时，可检查模型能力、任务难度和 GROUP_SIZE。

`πref` 是训练开始时加载、随后固定的参考模型；即使关闭 KL，也会创建并计算参考策略。
`old_log_probs` 仍由公共训练框架生成用于诊断，但 TAPO 损失的分母不会使用它。
熵是完整词表分布的熵，单位为 nats；0.01 是绝对阈值，使用严格 `>`，不是百分位数。
CE 仅作用于混合组中的成功回答，不做熵过滤。与 main 的公共超参数对齐后，默认添加
`low_var_kl` KL loss，系数为 0.01；可用 `actor_rollout_ref.actor.use_kl_loss=False` 关闭。
该项是 actor loss 的正则项，不是 reward shaping。

主要参数可通过命令行覆盖：

```bash
sbatch scripts/train_tapo.sh \
  actor_rollout_ref.actor.policy_loss.tapo_entropy_threshold=0.1 \
  actor_rollout_ref.actor.policy_loss.tapo_epsilon=0.2 \
  actor_rollout_ref.actor.policy_loss.tapo_ce_coef=0.0 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
```

`tapo_ce_coef=0` 关闭成功样本 CE；如需单独的 TAPO 项，还可设 `tapo_loss_coef=1.0`。
默认每任务 8 条轨迹、16 个训练任务和 128 个验证任务。修改采样规模使用环境变量：

```bash
TRAIN_SIZE=16 VAL_SIZE=128 GROUP_SIZE=8 sbatch --export=ALL scripts/train_tapo.sh
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


## 与 main 同步的输出管理与参数

运行名称为 `<algorithm>_<model basename>_<YYYYMMDD_HHMMSS>`，时间采用 Asia/Shanghai。
默认示例为 `tapo_Qwen2.5-7B-Instruct_20261115_123456`。名称中的算法和模型读取实际
训练脚本默认值、环境变量 MODEL_PATH 及命令行覆盖项（最后一个覆盖项生效）。
通过 `scripts/train_tapo.sh` 启动时，默认目录如下：

```text
outputs/
  <RUN_NAME>/
    .hydra/                         # Hydra 配置、运行配置、命令行覆盖项
    main_ppo.log                    # Hydra 主进程日志；完整 Slurm 输出另存
  wandb/<RUN_NAME>/                  # W&B SDK 本地运行文件，初始化失败时可能只有诊断文件
  tensorboard/<RUN_NAME>/
    events.out.tfevents.*.<run_id>   # 标量指标；run_id 为 UUID hex
    config.<run_id>.json            # 解析后的完整训练配置，与事件文件配对
    memory_snapshots/
      my_snapshot_<YYYYMMDD_HHMMSS_microseconds>_<host>_rank<rank>_pid<pid>_<stage>.pickle
checkpoints/nest/<RUN_NAME>/
  latest_checkpointed_iteration.txt
  global_step_<step>/
    actor/                         # 分片 model、optimizer、extra 和模型配置
    data.pt                        # dataloader 恢复状态；TAPO 不使用 critic
slurm-<jobid>.out                   # sbatch 默认的 stdout/stderr，含 Ray worker 输出
```

`WANDB_DIR` 和 `TENSORBOARD_DIR` 可由环境变量覆盖，并传入 Ray worker。
W&B 初始化使用 30 秒超时；SDK 初始化/认证错误或未安装时打印原因，自动启用
console 和 TensorBoard，验证样本不再调用未初始化的 W&B。连接成功时三个后端同时使用。
该回退仅处理初始化阶段，不负责训练中途断网后的动态切换，也不会补传回退期间数据。

CUDA 内存分配历史默认开启，最多保留 100000 条；每个 worker 的模型初始化、
rollout、actor update、策略及参考策略 log probability 阶段首次成功时保存快照，
这些阶段发生异常时也尝试保存。快照记录失败不覆盖原训练异常。可在启动前设置
`CUDA_MEMORY_HISTORY=0` 关闭；CPU 环境自动跳过。只有阶段实际执行且 CUDA API 可用
时才会产生快照，快照不是模型 checkpoint。

模型 checkpoint 默认每 25 个训练 step 及最终 step 保存；默认保存 model、optimizer、extra。
每 5 step 验证，训练前先验证，总计 150 epochs。`rollout_data_dir` 和
`validation_data_dir` 默认均为 null，不自动保存完整轨迹；`log_val_generations=0`。
TensorBoard 配置 JSON 包含 TAPO 专属参数。

与 main GiGPO 入口同步的主要值：Qwen2.5-7B-Instruct，4 GPU / TP=4，
train/val batch=16/128，group=8，PPO mini batch=256，actor/rollout/ref micro batch=8，
rollout max_num_seqs=128、gpu_memory_utilization=0.5、enforce_eager=False，
KL loss=True、coef=0.01、type=low_var_kl，gamma=0.95。TAPO 优势不使用 gamma。
学习率 1e-6、prompt/response 长度 2048/512、max_steps=50 等原本一致的参数保持一致。
Slurm CPU 从 16 调整为 32，以覆盖 25.6 个环境 worker CPU 预留和 GPU worker / TaskRunner。

保留 TAPO 的 estimator、policy loss、熵阈值、CE 权重、二元成功优势及
`filter_groups.enable=False`、`use_invalid_action_penalty=False`；开启非法动作 reward
惩罚会违反 TAPO 训练器的二元成功标签校验，因此不从 GiGPO 覆盖此开关。
