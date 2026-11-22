#!/usr/bin/env bash
#SBATCH --chdir=/data/home/zdhs0200/NEST/TAPO
#SBATCH --output=/data/home/zdhs0200/NEST/TAPO/slurm-%j.out
#SBATCH --error=/data/home/zdhs0200/NEST/TAPO/slurm-%j.out
#SBATCH --job-name=train_tapo
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Slurm GPU 数在提交时通过 --gres=gpu:N 指定。

#SBATCH --mem=256G
#SBATCH --partition=project3-1
#SBATCH --account=project3

set -euo pipefail
# Slurm executes a spool copy; use this worktree explicitly.
PROJECT_ROOT=${PROJECT_ROOT:-/data/home/zdhs0200/NEST/TAPO}
cd "$PROJECT_ROOT"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    job_output_dir="$PROJECT_ROOT/outputs/$SLURM_JOB_ID"
    mkdir -p "$job_output_dir"
    # Slurm opens stdout and stderr before the job can create its output directory.
    slurm_log="/data/home/zdhs0200/NEST/TAPO/slurm-$SLURM_JOB_ID.out"
    if [[ "$slurm_log" -ef /dev/stdout && "$slurm_log" -ef /dev/stderr ]]; then
        mv -T -- "$slurm_log" "$job_output_dir/slurm-$SLURM_JOB_ID.out"
    fi
fi
source /data/apps/apps/Miniconda3/latest/etc/profile.d/conda.sh
conda activate NEST

# 在计算节点上建立到登录节点代理的 SSH 隧道
ssh -4 -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
    -L 127.0.0.1:17890:127.0.0.1:7890 \
    zdhs0200@10.36.204.1 &
SSH_TUNNEL_PID=$!
trap 'kill "$SSH_TUNNEL_PID" 2>/dev/null || true' EXIT

# 等待隧道就绪；最多等 10 秒
TUNNEL_READY=0
for i in {1..20}; do
    if ! kill -0 "$SSH_TUNNEL_PID" 2>/dev/null; then
        echo "SSH 隧道启动失败，请检查 SSH 日志和代理端口" >&2
        exit 1
    fi
    if (: > /dev/tcp/127.0.0.1/17890) 2>/dev/null; then
        TUNNEL_READY=1
        break
    fi
    sleep 0.5
done

if [[ "$TUNNEL_READY" -ne 1 ]]; then
    echo "SSH 隧道未在 10 秒内就绪，停止训练" >&2
    exit 1
fi

export http_proxy=http://127.0.0.1:17890
export https_proxy=http://127.0.0.1:17890

bash "$PROJECT_ROOT/examples/tapo_trainer/run_alfworld.sh" vllm "$@"
