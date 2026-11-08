#!/usr/bin/env bash
#SBATCH --job-name=train_tapo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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
bash examples/tapo_trainer/run_alfworld.sh vllm \
    'hydra.run.dir=outputs/tapo/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
    hydra.job.chdir=False "$@"
