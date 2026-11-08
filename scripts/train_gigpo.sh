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
exec bash -e "$PROJECT_ROOT/examples/gigpo_trainer/run_${ENV_NAME}.sh" vllm \
  'hydra.run.dir='"$PROJECT_ROOT"'/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
  hydra.job.chdir=False \
  trainer.project_name=nest \
  trainer.experiment_name="gigpo_${ENV_NAME}" \
   "$@"


#sbatch -N 1 --gres=gpu:2 --mem=256G -p project3-1 -A project3 scripts/train_gigpo.sh
