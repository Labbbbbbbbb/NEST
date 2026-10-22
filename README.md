# NEST

面向多轮 LLM Agent 细粒度信用分配研究的精简工作仓库，基于本地 verl-agent 工作树创建。

## 当前能力与边界

- GiGPO：保留原实现、环境交互、奖励转换、训练调度和更新链路。
- GRPO：保留作为同环境、同预算的对照基线。
- ARPO / TAPO：尚未实现
- 本次是工程整理，不是论文结果复现；未进行 GPU 训练验证。
- 保留 `verl`、`agent_system`、`gigpo` 的 Python 导入名，避免破坏原有动态导入和检查点工具。

## 从哪里读起

1. `verl/trainer/main_ppo.py`：组装环境、数据集、奖励管理器和远程 worker。
2. `verl/trainer/ppo/ray_trainer.py`：`fit()` 调度采样、打分、优势与参数更新。
3. `agent_system/multi_turn_rollout/rollout_loop.py`：环境交互和 step 展开。
4. `agent_system/reward_manager/episode.py`：环境/轨迹奖励到 token 奖励。
5. `gigpo/core_gigpo.py`：折扣回报、观察分组和联合优势。
6. `verl/workers/actor/`：消费优势，计算策略损失、更新参数。

## 安装与运行

训练面向 Linux + NVIDIA GPU；Windows 工作目录用于编辑。需要模型、环境数据和适配 CUDA 的运行环境。
沿用原仓库 README 的安装基线（不是本机验证过的锁定环境）：

```bash
conda create -n nest python=3.12 -y
conda activate nest
pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
alfworld-download -f
```

首次验证 ALFWorld GiGPO 链路（会下载数据/模型并实际使用 GPU）：

```bash
bash scripts/train_gigpo.sh alfworld trainer.total_training_steps=1 trainer.val_before_train=False trainer.test_freq=-1
```

正式运行时去掉上述 smoke 覆盖项，并按机器资源配置 batch、GPU、显存和训练步数。
这不是“单步运行即复现论文”；原示例已有本地小显存改动，见 SOURCE_MANIFEST.json。
默认日志为 console；需要 wandb 时显式覆盖 trainer.logger。

```bash
bash scripts/train_gigpo.sh alfworld trainer.total_epochs=150
# 同环境的 GRPO 对照，注意其他预算需保持一致：
bash scripts/train_gigpo.sh alfworld algorithm.adv_estimator=grpo
# 源示例仍可直接使用：
bash examples/gigpo_trainer/run_alfworld_lora.sh vllm
```

数据准备脚本沿用上游：ALFWorld 的 text parquet 主要提供占位输入及样本数量，实际任务来自环境 reset；Search 使用真实查询数据，不能混用占位数据。数据准备可能访问 Hugging Face。
WebShop、Search 和视觉环境需要额外安装与服务，见 `docs/UPSTREAM_README.md` 的对应章节；该文档保留原文，部分原有图片或示例链接已随精简移除。

## 精简策略

已移除其他 recipe、无关算法示例、原文档站点/图片、Docker/NPU 辅助配置及大部分上游测试。
保留完整 `verl/` 训练底座与 `agent_system/` 环境适配层：它们包含条件导入和共享依赖，暂不逐文件裁剪。
因此这是可继续开发的保守精简版，不是只剩几个算法文件的完全重写版。
保留协议测试 `tests/test_protocol.py`；未将上游所有测试视为本工程必需依赖。

`pyproject.toml` / `setup.py` 是安装依赖来源。没有复制旧的 requirements.txt：它和 setup.py 的 tensordict、transformers 约束冲突。
项目显示名为 NEST，发行包名和导入名仍兼容上游 verl；请使用独立虚拟环境，避免与另一份 verl 混装。

## 验证与溯源

```bash
python scripts/check_source.py
python -m compileall -q verl agent_system gigpo
pytest tests/test_protocol.py -q
```

前两个检查不需要训练依赖，但需要 Python；协议测试需要安装依赖。
`check_source.py` 验证导出文件哈希、Python 语法、关键入口和排除目录，不代替 GPU 集成测试。
`SOURCE_MANIFEST.json` 记录原提交、逐文件哈希、排除文件；继承原许可证和各源文件版权。
当前未安装 Python/GPU 训练环境，创建时仅执行 PowerShell 文件哈希、目录完整性和 git 差异检查。

扩展规划与方法身份说明见 `docs/RESEARCH_MAP.md`。
