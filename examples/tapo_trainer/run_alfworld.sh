#!/usr/bin/env bash
set -euo pipefail
ENGINE=${1:-vllm}
if [[ $# -gt 0 ]]; then shift; fi
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

# 8*8 train actors + 16 validation actors reserve 8 logical CPUs at 0.1 each.
# Two GPU actors + TaskRunner also need CPUs; request 16 CPUs in Slurm.
TRAIN_SIZE=${TRAIN_SIZE:-8}
VAL_SIZE=${VAL_SIZE:-16}
GROUP_SIZE=${GROUP_SIZE:-8}
N_GPUS=${N_GPUS:-2}
RAY_CPUS=${RAY_CPUS:-${SLURM_CPUS_PER_TASK:-16}}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
python3 -m examples.data_preprocess.prepare --mode text \
    --train_data_size "$TRAIN_SIZE" --val_data_size "$VAL_SIZE"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=tapo \
    algorithm.filter_groups.enable=False \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=tapo \
    +actor_rollout_ref.actor.policy_loss.tapo_entropy_threshold=0.01 \
    +actor_rollout_ref.actor.policy_loss.tapo_epsilon=0.2 \
    +actor_rollout_ref.actor.policy_loss.tapo_loss_coef=0.8 \
    +actor_rollout_ref.actor.policy_loss.tapo_ce_coef=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_kl_loss=False \
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
    actor_rollout_ref.model.path="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$N_GPUS" \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n="$GROUP_SIZE" \
    env.resources_per_worker.num_cpus=0.1 \
    ray_init.num_cpus="$RAY_CPUS" \
    trainer.critic_warmup=0 \
    'trainer.logger=[console]' \
    trainer.project_name=nest \
    trainer.experiment_name=tapo_alfworld \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    "$@"
