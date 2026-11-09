#!/usr/bin/env bash
#SBATCH --job-name=train_tapo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --partition=project3-1
#SBATCH --account=project3

set -euo pipefail
# Slurm executes a spool copy; resolve the repository through the submission dir.
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/data/home/zdhs0200/NEST}}
cd "$PROJECT_ROOT"
source /data/apps/apps/Miniconda3/latest/etc/profile.d/conda.sh
conda activate NEST

# Reuse exported http_proxy/https_proxy if downloads require your login proxy.
# No tunnel is started automatically; fully cached models can run offline.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
# 从命令行覆盖项取得算法和模型；与 Hydra 一样，最后一个覆盖项生效。
ALGORITHM="tapo"
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


bash examples/tapo_trainer/run_alfworld.sh vllm \
    'hydra.run.dir=outputs/tapo/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
    hydra.job.chdir=False \
    trainer.project_name=nest \
    trainer.experiment_name="$RUN_NAME" \
    'trainer.logger=[console,wandb,tensorboard]' \
    "++ray_init.runtime_env.env_vars.TENSORBOARD_DIR=$TENSORBOARD_DIR" \
    "++ray_init.runtime_env.env_vars.WANDB_DIR=$WANDB_DIR" \
    "$@"



#只需要运行：sbatch scripts/train_tapo.sh

# 验证：
# sbatch scripts/train_tapo.sh \
#   trainer.total_training_steps=1 \
#   trainer.val_before_train=False