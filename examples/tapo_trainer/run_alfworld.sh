#!/usr/bin/env bash
set -euo pipefail
# 只修改这一行选择模型，相关参数均在本文件内设置。
MODEL_SIZE="1.5B"  # "7B" 或 "1.5B"
case "$MODEL_SIZE" in
  7B)
    MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
    N_GPUS=4
    TP_SIZE=4
    PPO_MICRO_BATCH_SIZE=8
    ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE=8
    REF_LOG_PROB_MICRO_BATCH_SIZE=8
    GPU_MEMORY_UTILIZATION=0.5
    ;;
  1.5B)
    MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
    N_GPUS=2
    TP_SIZE=2
    PPO_MICRO_BATCH_SIZE=32
    ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE=32
    REF_LOG_PROB_MICRO_BATCH_SIZE=32
    GPU_MEMORY_UTILIZATION=0.6
    ;;
  *) echo "MODEL_SIZE 只能是 7B 或 1.5B" >&2; exit 2 ;;
esac
PPO_MINI_BATCH_SIZE=256
MAX_NUM_SEQS=128
ACTOR_PARAM_OFFLOAD=False
ACTOR_OPTIMIZER_OFFLOAD=False
REF_PARAM_OFFLOAD=True
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
ENGINE=${1:-vllm}
if [[ $# -gt 0 ]]; then shift; fi
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

# Match main GiGPO sampling and GPU defaults; retain TAPO-specific loss settings.
# CPU allocation follows the scheduler; RAY_CPUS is an optional explicit override.
TRAIN_SIZE=${TRAIN_SIZE:-16}
VAL_SIZE=${VAL_SIZE:-128}
GROUP_SIZE=${GROUP_SIZE:-8}
RAY_CPUS=${RAY_CPUS:-null}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
python3 -m examples.data_preprocess.prepare --mode text \
    --train_data_size "$TRAIN_SIZE" --val_data_size "$VAL_SIZE"

# 日志名称跟随实际模型；命令行模型覆盖仍可使用。
LOG_MODEL_PATH="$MODEL_PATH"
for arg in "$@"; do
    case "$arg" in
        actor_rollout_ref.model.path=*|+actor_rollout_ref.model.path=*|++actor_rollout_ref.model.path=*) LOG_MODEL_PATH="${arg#*=}" ;;
    esac
done
MODEL_NAME="${LOG_MODEL_PATH%/}"
MODEL_NAME="${MODEL_NAME##*/}"
RUN_NAME="tapo_${MODEL_NAME//[^a-zA-Z0-9._-]/_}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
export WANDB_DIR="$PROJECT_ROOT/outputs/wandb/$RUN_NAME"
export TENSORBOARD_DIR="$PROJECT_ROOT/outputs/tensorboard/$RUN_NAME"
mkdir -p "$WANDB_DIR" "$TENSORBOARD_DIR"
echo "Model: $MODEL_PATH; GPUs: $N_GPUS; micro batch: $PPO_MICRO_BATCH_SIZE; run: $RUN_NAME"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=tapo \
    algorithm.filter_groups.enable=False \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    actor_rollout_ref.actor.policy_loss.loss_mode=tapo \
    +actor_rollout_ref.actor.policy_loss.tapo_entropy_threshold=0.01 \
    +actor_rollout_ref.actor.policy_loss.tapo_epsilon=0.2 \
    +actor_rollout_ref.actor.policy_loss.tapo_loss_coef=0.8 \
    +actor_rollout_ref.actor.policy_loss.tapo_ce_coef=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    data.train_files="$HOME/data/verl-agent/text/train.parquet" \
    data.val_files="$HOME/data/verl-agent/text/test.parquet" \
    data.train_batch_size="$TRAIN_SIZE" \
    data.val_batch_size="$VAL_SIZE" \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD" \
    actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOG_PROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n="$GROUP_SIZE" \
    env.resources_per_worker.num_cpus=0.1 \
    ray_init.num_cpus="$RAY_CPUS" \
    trainer.critic_warmup=0 \
    'trainer.logger=[console,wandb,tensorboard]' \
    trainer.project_name=nest \
    trainer.experiment_name="$RUN_NAME" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    "hydra.run.dir=$PROJECT_ROOT/outputs/$RUN_NAME" \
    hydra.job.chdir=False \
    "++ray_init.runtime_env.env_vars.WANDB_DIR=$WANDB_DIR" \
    "++ray_init.runtime_env.env_vars.TENSORBOARD_DIR=$TENSORBOARD_DIR" \
    "$@"
