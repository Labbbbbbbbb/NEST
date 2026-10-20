# 研究模块与复现边界

## 相对通用 veRL 底座的 Agent 扩展

这里按当前本地代码职责说明，不声称完成了与某个 veRL 历史提交的逐行差异比较，也不意味着最新 veRL 没有多轮功能。

| 模块 | 功能 | 对细粒度信用分配的作用 |
| --- | --- | --- |
| agent_system/environments | reset/step、并行环境、组初始状态、动作有效性、观察锚点 | 同起点对比及环境奖励 |
| agent_system/memory | 历史管理、每步上下文构建 | 控制模型每步能看到的信息 |
| agent_system/multi_turn_rollout | step-independent 交互、轨迹采集、动态采样、批次调整 | 保留 step 与 trajectory 的关联 |
| agent_system/reward_manager | episode 评分与 token 奖励载体 | 奖励信号进入学习流程 |
| gigpo/core_gigpo.py | 轨迹内折扣回报、轨迹组与观察组优势 | episode advantage + weight * step advantage |
| verl/trainer/main_ppo.py、ppo/ray_trainer.py | 将上述组件接到 Ray/模型训练 | 完整实验管线 |

保留的 `verl/` 包括 Ray 调度、DataProto、FSDP、推理后端、策略优化、检查点与工具函数。这些并非无用模块；为了避免可选分支和动态导入失效，初版不拆开它们。

## 算法并不是只改 reward 函数

- 环境 reward：行为得到的原始反馈。
- return：从当前 step 起的未来奖励累计/折扣累计。
- advantage：相对组或基线的优劣，直接参与策略目标。
- loss mask / weight：哪些 token 被更新、如何加权。
- rollout 分支：在哪些状态额外采样，改变能用于比较的数据。

GiGPO 的关键是 advantage 构造。其 step_rewards 字段在主线中实际承载折扣回报。
修改奖励、修改优势、修改 token 掩码、修改采样树是不同层级，不能一概包装为“替换奖励函数”。

## 已确认的目标方法

仓库没有 ARPO/TAPO 实现，暂不注册同名空算法，也不把其他算法重命名冒充复现。

用户已确认的官方来源：
- Agentic Reinforced Policy Optimization：https://arxiv.org/abs/2507.19849 ，官方实现 https://github.com/RUC-NLPIR/ARPO 。其自适应分支采样意味着可能需要改 collector 和环境恢复/重放机制，不能只替换优势函数。
- Token-level Advantage Policy Optimization from Negative Feedback in Multi-Turn Agents：https://github.com/Sunrepe/TAPO 。该方法基于 OpenRLHF，需要移植并核对负反馈、token 优势和熵筛选的完整链路。
- 已排除其他同名 ARPO/TAPO；不会引入 ARPO 仓库中的 AEPO 作为替代。

后续接入建议：先固定环境/模型/预算与 GRPO 对照，再根据论文实际需求扩展 collector、优势估计和 actor loss；复现必须检查采样策略、奖励规则、分组规则、token 对齐、归一化和评估方式。

## 当前代码需关注的既有边界

- adjust_batch 默认可能复制 step 补齐批次，重复行可能影响组统计，研究时应审计。
- 环境 reward、无效动作惩罚、KL 奖励和 actor KL loss 应分开记录。
- response_mask 只是 response 区间 attention_mask，不能默认当作所有算法的 token 学习掩码。
- 当前环境日志按 task / step / trajectory 的统计粒度不同，需要核对比较指标。
- 源 README 明确说后续更新改变过 GiGPO 表现；精确复现原论文应锁定其指定历史版本。
- 原有中文注释和本地实验参数随工作树复制，来源提交号不代表内容逐字等于该提交。

## 对接位置（基于官方 README 与本地架构的工程判断，尚未移植代码）

| 方法 | 需要保留/扩展的位置 | 不能省略的核对 |
| --- | --- | --- |
| GiGPO | collector、episode reward manager、core_gigpo | uid/traj_uid/anchor_obs、折扣及分组归一化 |
| RUC-NLPIR/ARPO | rollout 分支采样、工具执行、优势分配 | 高熵分支选择、共享前缀与分支归属、工具预算；不能只改 compute_advantage |
| Sunrepe/TAPO | 成功/失败轨迹数据管线、token 级优势、熵筛选与 actor loss | 官方实现基于 OpenRLHF；数据对齐和损失语义不能直接当作当前在线 PPO 的一个开关 |

因此保留底座中的 token 概率/熵计算与 mask 机制，但它们只是可复用能力，不表示上述算法已实现。