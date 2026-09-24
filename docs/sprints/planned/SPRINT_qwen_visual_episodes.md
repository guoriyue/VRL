# Qwen 视觉 episode：有界控制与明确的策略信用

状态：2026-09-22 的生产级引擎目标已授权实现。运行时、真实 controller 训练、
恢复和基线比较已打通；控制策略超过固定编辑的能力验收尚未通过。
本计划接替 parked agentic program 中已经删除的 Janus 起点。

## 实施顺序与现状

1. **已实现** Qwen Image 2.1 参考编辑、真实 reward 加权更新、checkpoint 恢复和
   完整数值 reward 分量。保留 one-shot GRPO 主路径。
2. **已实现** 位于单次生成之上的有界 episode：记录原始任务、当前图像、显式
   edit/stop 动作、动作似然、独立策略身份、资产谱系、调用成本和终局目标。
   工具失败不转成成功样本或零分样本。
3. **已实现** 独立 Qwen3-VL controller，观察原图与中间结果，在有限动作表中
   采样。编辑器保持冻结；固定路由和 best-of-N 都明确标为评估基线。
4. **已实现** controller 的真实 log-prob 回放、因果 return-to-go、组内其他
   episode 的 leave-one-out baseline，以及已有 clipped policy objective。
   controller 与 diffusion 轨迹和版本分别归属；stop 不伪造 diffusion action。
5. **进行中** 在来源分离的数据上比较 always-stop、固定一次编辑、随机动作、
   controller 和相同最大编辑预算的 best-of-N。报告实际调用量、额外判分开销、
   延迟、失败以及独立检查；不预设结果为正。
6. **后续门控项** 只有控制策略收益和归属/回放均通过后，再测试按实际编辑动作
   归因的 generator 交替更新。两个独立训练脚本不称作联合多策略 RL。

## 运行时契约

- episode 是有限时域任务，有明确编辑调用上限。预算终止属于这个有限目标；
  中断/失败轨迹不能进入策略更新。持续任务的 truncation bootstrap 另行设计。
- 每次决策观察前一步的实际产物。动作集合、预处理后的真实 tensor、观测摘要
  和策略版本都能回放。新记录还保存与 tensor 资产绑定的完整动作分布。
- 终局图片得分只支付一次；每次编辑在发生时支付成本。return-to-go 只包含
  当前之后的成本和终局结果；gamma 明确记录。
- episode 内策略版本不变。controller、generator、reward 各自有身份与所有权；
  当前不启用异步过期策略训练。
- 在 32 GB GPU 上按 controller、generator、judge 顺序切换。
  外部 judge 必须逐一证明设备隔离或支持确认释放的 parking；失败后不重用。
- reward 可独立使用 CPU verifier 或 HTTP 模型。目标图、遮罩等 reward-only
  资产记录内容摘要并校验，不作为 controller 图像或编辑器条件输入。
- trace 保存可序列化记录和资产引用，不保存活模型对象或 Ray handle。

## 已有证据与未通过项

- 真实 Qwen 加载、图像条件决策、旧/新似然回放、非零梯度和 optimizer 更新
  已验证；这不是 mock 证明。
- optimizer、RNG、episode cursor、基座内容身份、编辑器身份和 reward 配方
  进入 checkpoint/恢复契约。旧版无额外 reward 资产的身份保持兼容。
- Qwen factory 的基座身份 v2 还绑定实际 prompt 模板与 Pillow 版本。每个
  controller 持有不可变模板快照；模板变化会拒绝不兼容恢复。旧基座身份 v1
  的实验用其记录的冻结代码复现，不为旧 checkpoint 猜测缺失的模板身份。
- 初始八轮像素 verifier 训练完成 64 个 episode，所有更新回放误差为零。
  两个监测来源上的净回报从 0.6094 到 0.6199，仍低于固定一次编辑的 0.8040。
  不把这个小样本变化称为 agentic 能力提升。
- 扩大探索的实验已完成 512 个真实 episode、16 次更新，回放误差均为零。
  16 个独立最终来源、每 checkpoint 64 次比较中，平均净回报从 0.65948 到
  0.69305；来源配对差值区间 [-0.01789, 0.08348] 跨零。16 个来源均低于
  固定一次编辑的 0.80530。已完成状态反而更频繁地被重复编辑，能力门未通过。
- 默认及当前冻结实验仍使用原图/当前图的 RGB 合成视图。新增可选 `rgba`
  观测模式，额外提供两张实际 alpha 蒙版；模式绑定回放与 checkpoint，不暴露
  oracle 目标图。四图观测的真实模型与能力验证另行记录，不能从接口支持推断收益。
- 同设置的 alpha 观测训练与配对评估已预先排队。另建 RGB 合成图完全相同、
  仅 alpha 不同的状态对；目前只查询训练来源。真实 CPU Qwen 能在单图及四图
  输入下正确分类 8/8 个 mask 状态，但三种编辑/停止说明仍未建立可靠的条件
  停止。未据此新增菜单 API 或宣称 prompt 改写就是 RL 能力提升。
- diffusion 路径的独立 RGBA 提升证据，不能替代 controller 能力验收。

## 架构边界

改变：加入 episode 调度/信用分配边界、具体 controller/tool adapter 和可验证
的 reward-only 资产；独立重打分、诊断和校准仍留在 reward 子系统。

保留：GenerationRuntime 的单次调用 API、已有 family encode/prepare/replay
形状、TrajectoryBatch 的单策略含义，以及 one-shot collector/trainer 快路径。
controller/tool 的薄 adapter 承担策略与资源生命周期边界，有保留价值。
任务动作指令属于 manifest；schema 名称和模型维度可以是协议常量，不在
执行代码中塞入大段 ALL_CAPS 业务词表。

非目标：恢复已删除的 token-AR family、任意外部工具、新 serving scheduler、
两套策略同时更新，以及仅为了减少行数而拆分/合并文件。

运行方式见 [visual_controller_rl.md](../../visual_controller_rl.md)；完整正负结果和
实验身份见 [工作台账](../../reports/visual_rl_engine_20260922/PROGRESS.md)。
