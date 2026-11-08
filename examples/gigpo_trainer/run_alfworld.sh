set -x
ENGINE=${1:-vllm}
if [[ $# -gt 0 ]]; then
    shift
fi
# export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  #changed by zyt

                            #0.1
num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=16  #16  ,这两个数太大容易把前面负责rollout的worker搞deid掉，下面ppo_micro_batch_size_per_gpu太大容易导致GPU OOM （原脚本是32）
val_data_size=128   #128
group_size=8
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size



# 上面修改过的参数都有注释，下面因为\后面不给注释，所以修改和参数说明统一在这里面注释说明
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \  actor训练时每张gpu每次前向传播的样本数，太大容易导致GPU OOM。
#                                                                 它与mini不同，mini是一次训练的总样本数，micro是每张gpu每次前向传播的样本数，
#                                                                 mini/micro=总样本数/每张gpu每次前向传播的样本数=每次训练的总前向传播次数。
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \  重新计算旧策略 log probability 的分块大小，过大也会 OOM。
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \    这里是ref的模型，用于计算KL散度，与上一行的old policy log prob不是同一个，但分块的原理是一样的
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \   传给 vLLM 的显存预算比例。一般来说，设得越大，越容易让同卡上的【其他】计算发生 OOM；设得太小，则可能导致 vLLM 的 KV cache 空间不足、并发下降，甚至初始化失败。
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \ 控制 actor 训练时，是否把模型参数（权重）卸载到 CPU 内存，以节省 GPU 显存
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \  控制 actor 训练时，是否把优化器状态卸载到 CPU 内存，以节省 GPU 显存。True时节约显存，但增加内存占用以及CPU-GPU通信开销
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \ 反向传播时重算部分激活，降低训练激活显存，增加计算量
#     actor_rollout_ref.model.use_remove_padding=True \ 前向计算移除 padding，减少无效 token 的计算和中间张量
#     rollout.tensor_model_parallel_size=1 \  推理张量并行度。多卡时可分摊模型和部分推理资源；当前单卡就是 1
#     rollout.enforce_eager=False \ 允许 CUDA Graph；通常有额外显存开销，具体依赖 vLLM 版本与执行路径
#     rollout.enable_chunked_prefill=False \ 控制长 prompt 的 prefill 调度，影响推理峰值和吞吐，不直接改变奖励公式
#     group_size=8 → env.rollout.n=8  \ 每个任务运行 8 条轨迹，增加环境 worker 数、rollout 数据量，同时改变 GiGPO 的组内比较,【不能随便改小】
#     data.max_prompt_length=2048	\ 每一步模型输入的长度限制，影响 attention、KV cache 和训练激活
#     data.max_response_length=512	\ 每一步生成的最大 token 数，不是整条 50 步轨迹的总长度
#     trainer.n_gpus_per_node=1、nnodes=1	决定并行规模、资源池以及部分 batch 的归一化和整除要求
#     env.resources_per_worker.num_cpus=num_cpus_per_env_worker=1 

#     ============================以上基本为直接影响显存的参数，调训练效果还要看另外的如奖励系数、学习率等等===============================



python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='gigpo_qwen2.5_1.5b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
        $@