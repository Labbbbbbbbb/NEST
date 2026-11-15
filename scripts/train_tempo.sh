#!/usr/bin/env bash
#SBATCH --job-name=train_tempo
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Slurm GPU 数在提交时通过 --gres=gpu:N 指定。

#SBATCH --mem=256G
#SBATCH --partition=project3-1
#SBATCH --account=project3

set -euo pipefail
# Slurm executes a spool copy; use this worktree explicitly.
PROJECT_ROOT=${PROJECT_ROOT:-/data/home/zdhs0200/NEST/TEMPO}
cd "$PROJECT_ROOT"
source /data/apps/apps/Miniconda3/latest/etc/profile.d/conda.sh
conda activate NEST
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Reuse exported http_proxy/https_proxy if downloads require your login proxy.
# No tunnel is started automatically; fully cached models can run offline.
bash "$PROJECT_ROOT/examples/tempo_trainer/run_alfworld.sh" vllm "$@"
