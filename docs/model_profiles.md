# 模型选择

仅在 examples/tapo_trainer/run_alfworld.sh 修改 MODEL_SIZE="7B" 或 "1.5B"。同文件case分支定义模型、GPU/TP、三个micro batch和vLLM显存比例。7B为4卡/micro8/0.5；1.5B为2卡/micro32/0.6。1.5B取自原示例，未做GPU验证。

提交方式：sbatch --gres=gpu:4 scripts/train_tapo.sh（GiGPO原有包装还可加alfworld参数）；1.5B对应--gres=gpu:2。Slurm在运行脚本前分配资源，MODEL_SIZE不改变已分配的GPU。无外部预设文件或额外提交入口。日志由run_alfworld.sh按所选模型命名。
