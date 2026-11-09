#!/usr/bin/env bash
# NEST entry point; the original GiGPO implementation remains unchanged.

set -euo pipefail

PROJECT_ROOT="/data/home/zdhs0200/NEST"
cd "$PROJECT_ROOT"

source /data/apps/apps/Miniconda3/latest/etc/profile.d/conda.sh
conda activate NEST

# 清除 AMD GPU 可见性变量，保留 Slurm 分配的 CUDA_VISIBLE_DEVICES
# 应对报错：ValueError: Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# 在计算节点上建立到登录节点代理的 SSH 隧道
ssh -4 -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
    -L 127.0.0.1:17890:127.0.0.1:7890 \
    zdhs0200@10.36.204.1 &
SSH_TUNNEL_PID=$!
trap 'kill "$SSH_TUNNEL_PID" 2>/dev/null || true' EXIT

# 等待隧道就绪；最多等 10 秒
for i in {1..20}; do
    if ! kill -0 "$SSH_TUNNEL_PID" 2>/dev/null; then
        echo "SSH 隧道启动失败，请检查 SSH 日志和代理端口" >&2
        exit 1
    fi
    if (: > /dev/tcp/127.0.0.1/17890) 2>/dev/null; then
        break
    fi
    sleep 0.5
done

export http_proxy=http://127.0.0.1:17890
export https_proxy=http://127.0.0.1:17890

# sbatch 会将脚本复制到 Slurm spool，不能用 BASH_SOURCE 推算项目目录。
# 保持开头 cd "$PROJECT_ROOT" 设置的工作目录。
ENV_NAME="${1:-alfworld}"
if [[ $# -gt 0 ]]; then shift; fi
case "$ENV_NAME" in
  alfworld|webshop|sokoban) ;;
  *) echo "Supported wrapper environments: alfworld, webshop, sokoban" >&2; exit 2 ;;
esac

# 与各环境启动脚本中的默认模型保持一致。
case "$ENV_NAME" in
  sokoban) MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct" ;;
  *) MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct" ;;
esac

# 从命令行覆盖项取得算法和模型；与 Hydra 一样，最后一个覆盖项生效。
ALGORITHM="gigpo"
for arg in "$@"; do
  case "$arg" in
    algorithm.adv_estimator=*|+algorithm.adv_estimator=*|++algorithm.adv_estimator=*)
      ALGORITHM="${arg#*=}"
      ALGORITHM="${ALGORITHM//\'/}"
      ALGORITHM="${ALGORITHM//\"/}"
      ;;
    actor_rollout_ref.model.path=*|+actor_rollout_ref.model.path=*|++actor_rollout_ref.model.path=*)
      MODEL_PATH="${arg#*=}"
      MODEL_PATH="${MODEL_PATH//\'/}"
      MODEL_PATH="${MODEL_PATH//\"/}"
      ;;
  esac
done
# 仓库 ID 和本地路径均取最后一级作为模型名。
MODEL_NAME="${MODEL_PATH%/}"
MODEL_NAME="${MODEL_NAME##*/}"
RUN_NAME="${ALGORITHM//[^a-zA-Z0-9_-]/_}_${MODEL_NAME//[^a-zA-Z0-9._-]/_}_$(date +%Y%m%d_%H%M%S)"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_ROOT/outputs/wandb/$RUN_NAME}"
mkdir -p "$WANDB_DIR"
echo "W&B 实验名称：$RUN_NAME"
echo "W&B 本地日志目录：$WANDB_DIR"

# 同时记录到控制台、W&B 和服务器本地 TensorBoard；每次运行独立保存。
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$PROJECT_ROOT/outputs/tensorboard/$RUN_NAME}"
mkdir -p "$TENSORBOARD_DIR"
echo "TensorBoard 日志目录：$TENSORBOARD_DIR"

exec bash -e "$PROJECT_ROOT/examples/gigpo_trainer/run_${ENV_NAME}.sh" vllm \
  'hydra.run.dir='"$PROJECT_ROOT"'/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
  hydra.job.chdir=False \
  trainer.project_name=nest \
  trainer.experiment_name="$RUN_NAME" \
  'trainer.logger=[console,wandb,tensorboard]' \
  "++ray_init.runtime_env.env_vars.TENSORBOARD_DIR=$TENSORBOARD_DIR" \
  "++ray_init.runtime_env.env_vars.WANDB_DIR=$WANDB_DIR" \
   "$@"


#sbatch -N 1 --gres=gpu:2 --mem=256G -p project3-1 -A project3 scripts/train_gigpo.sh
