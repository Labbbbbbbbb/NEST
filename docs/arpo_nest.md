# NEST ARPO 工程与运行说明

本工作区从 main 的 8ada30a 创建，沿用 GiGPO 的 verl/ALFWorld 框架。

## 启动

```bash
cd /data/home/zdhs0200/NEST/ARPO
sbatch scripts/train_arpo.sh
# 首次 GPU 验证建议只执行一次更新：
sbatch scripts/train_arpo.sh trainer.total_training_steps=1 trainer.val_before_train=False
```

默认使用 NEST conda 环境，4 GPU、256G 内存，CPU 数由 Slurm 分配，脚本不固定申请32个CPU；模型 Qwen/Qwen2.5-7B-Instruct，训练任务16、验证任务128、每任务8条轨迹，prompt/response上限2048/512，每条轨迹最多50步；学习率1e-6，mini batch256，micro batch每卡8，vLLM显存利用率0.5。验证每5步、保存每25步，共150 epoch。通用配置对齐现有 GiGPO/TAPO；算法采用结果奖励和 actor KL loss(0.01)，禁用奖励KL、无效动作奖励惩罚和额外熵损失。gamma保留0.95，但本算法结果优势不使用逐步折扣。

## 目录职责

| 目录 | 功能 |
|---|---|
| scripts/ | Slurm启动、环境激活、工作区路径和日志目录；train_arpo.sh 是本分支入口 |
| examples/arpo_trainer/ | run_alfworld.sh 定义训练参数和命令行覆盖 |
| examples/data_preprocess/ | 准备数据占位记录，实际任务由ALFWorld提供 |
| arpo/ | 本算法的核心实现 |
| agent_system/environments/ | 环境构建、动作解析、提示模板、环境包 |
| agent_system/memory/ | 保存各环境的观测和动作历史，拼接下一回合输入 |
| agent_system/multi_turn_rollout/ | 模型生成与环境交互循环，整理逐回合训练行 |
| agent_system/reward_manager/ | 将完整episode奖励映射到训练样本 |
| verl/trainer/ | 配置、Ray训练调度、奖励与优势计算、评估、保存 |
| verl/workers/ | GPU上的actor/ref/critic、生成后端、FSDP与参数切换 |
| verl/models/ | 模型适配实现 |
| verl/single_controller/ | Ray worker与资源池管理 |
| verl/utils/ | 数据集、张量、日志、检查点等共用工具 |
| gigpo/ | 保留的GiGPO实现，供原有算法路径使用；本入口选择ARPO |
| tests/arpo/ | 算法公式与训练数据管线检查 |
| docs/ | 本适配说明与继承的框架文档 |

## 算法接入

arpo/rollout.py 实现训练时的自适应分叉；arpo/core_arpo.py 实现熵统计、分叉概率和轨迹级组内优势。环境 snapshot/restore/fork 在 agent_system/environments/env_package/alfworld/envs.py 中，通过同一任务的动作前缀重放恢复状态并核对观测。默认每组预算 8 条轨迹、初始 4 条、beam_size=2、branch_probability=0.5、entropy_weight=0.2。仅支持文本 ALFWorld + vLLM，domain_randomization=False。验证仍使用普通独立采样。

调用链：scripts/train_arpo.sh → examples/arpo_trainer/run_alfworld.sh → verl/trainer/main_ppo.py → verl/trainer/ppo/ray_trainer.py → TrajectoryCollector → 算法优势 → verl/workers/actor/dp_actor.py 更新策略。

NEST一行表示一个assistant回合，一条轨迹有多行。优势在补齐、裁剪和负载重排之前根据完整轨迹计算，之后跟随张量一起补齐和重排。保留原来的PPO裁剪策略损失与actor KL约束，不训练独立critic。

## 日志与数据

console、W&B和TensorBoard同时启用，项目名nest。run名为算法_模型_北京时间时间戳。输出分别位于 outputs/<run>/、outputs/wandb/<run>/、outputs/tensorboard/<run>/；检查点默认 checkpoints/nest/<run>/。数据占位文件与 GiGPO/TAPO 一致，默认位于 $HOME/data/verl-agent/text/，不属于训练输出；可通过 DATA_DIR 覆盖。

两个启动脚本显式设置当前工作区PYTHONPATH并传给Ray，避免引用旧的editable安装路径。模型下载所需代理沿用外部环境，不自动建立SSH隧道。DATA_DIR、MODEL_PATH、TRAIN_SIZE、VAL_SIZE、GROUP_SIZE、N_GPUS、RAY_CPUS可通过环境变量设置；RAY_CPUS默认null，由Ray自动识别资源（调度资源仍以Slurm为准），Hydra参数可追加到启动命令。

## 检查

```bash
PYTHONPATH=. python -m unittest discover -s tests/arpo -v
SKIP_DATA_PREP=1 bash examples/arpo_trainer/run_alfworld.sh vllm --cfg job
```

CPU测试与Hydra配置检查不代表GPU训练已通过；尚未提交GPU作业。SOURCE_MANIFEST.json 和 scripts/check_source.py 是继承的历史导出完整性记录，不能用其原始哈希验证本次算法修改。改动尚未提交到Git。

## 来源

https://github.com/RUC-NLPIR/ARPO/tree/e484f1ddc7b158acbc9c8cf5d03e8e2ccc6b4349/ARPO

本实现是移植到NEST多轮ALFWorld的适配，未复现上游数据集或报告的训练指标。
