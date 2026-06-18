# 技术路线重评：否定旧路线，转向最小可验证改进

日期：2026-06-18

本文档用于覆盖此前的技术路线。结论很明确：不要再把项目主线放在“言行不一致修复”、LoRA 微调或 RLHF 上。它们可以作为背景和辅助分析，但不适合作为当前项目的核心贡献。

## 当前事实

项目已经完成了 Alpamayo 1.5 的基础复现链路：

- 能在 PhysicalAI-AV 子集上跑 Alpamayo 1.5 baseline inference。
- 已经建立 named run 输出目录，避免实验结果混乱。
- 已经有 ADE/FDE、速度、加速度、jerk、reasoning-action consistency 等评测脚本。
- 已经有 failure mining report，能定位 stop/yield 等高风险失败样本。

这些工作有价值，但如果项目最终只停留在“复现 + 测评 + 失效分析”，面试表达会偏弱：它证明工程能力和研究判断，但缺少一点“我让系统变好了”的变化。

因此下一步需要一个很小但可验证的改进模块。

## 明确否定的旧路线

### 1. 不再以“言行不一致修复”为主线

否定原因：

- 这个指标依赖文本 intent 解析，规则脆弱，容易出现 false positive。
- stop/yield/slow_down 的边界不是天然清晰，阈值选择有主观性。
- 如果只优化这个指标，很容易变成“刷自定义 metric”，不一定代表驾驶行为真的更好。
- 它适合做 diagnostic signal，不适合作为项目核心贡献。

保留方式：

- 保留 reasoning-action consistency 作为辅助指标。
- 在报告里诚实说明：naive language-action consistency 存在误判，需要结合轨迹、速度曲线和视觉证据解释。

### 2. 不再以 LoRA / SFT 微调 Alpamayo 为主线

否定原因：

- Alpamayo 1.5 是 10B 级 VLA，训练成本高，面试项目里很难做出可信提升。
- 当前没有高质量人工修正标签，也没有稳定的任务级 reward。
- 小规模 LoRA 很容易只是在 60 个 val clips 上过拟合，泛化价值弱。
- 即使跑通，提升也很难和数据采样、随机性、阈值变化区分开。

保留方式：

- 暂时不做训练。
- 后续若必须有学习模块，可以只做一个 tiny reranker/calibrator，而不是动 Alpamayo 本体。

### 3. 不再以 RLHF / 强化学习微调为主线

否定原因：

- 官方 RL post-training 路线本身需要多卡大显存环境，超出当前项目体量。
- 缺少真实闭环环境或可靠 pseudo-closed-loop reward 时，RL 结论很难站住。
- reward hacking 风险高，面试中也容易被追问“到底优化了什么”。

保留方式：

- 可以在 related work 或 future work 中提到 RL，但当前不实现。

### 4. 不再以重型世界模型接入为主线

否定原因：

- “CoT 条件世界模型生成未来 BEV，再反喂 Alpamayo 修正决策”是一个完整研究项目，不是当前时间和资源下适合落地的面试项目。
- 需要额外模型、数据表征、训练目标和验证闭环，工程面过大。

保留方式：

- 可以保留 world-model-inspired 的思想：用未来轨迹、速度曲线和场景证据做 failure audit。
- 不声称训练或接入完整世界模型。

## 新主线：Inference-Time Trajectory Improvement

新的主线不是“训练 Alpamayo”，而是在推理阶段加一个轻量模块：

> Alpamayo 生成多个候选轨迹；小模块根据 CoC/intent 和轨迹动力学特征，选择或轻微修正更合理的轨迹。

推荐名称：

> Reasoning-Aware Trajectory Reranker for Alpamayo 1.5

这条路线的核心好处：

- 不需要训练 10B 模型。
- 不需要下载新的大数据。
- 不依赖闭环仿真。
- 能和已有 baseline / metrics / failure mining 直接衔接。
- 最后可以有一个明确对比：baseline first-sample vs best-of-N oracle vs reasoning-aware reranker。

## 具体方案

### Step 1: 多候选轨迹 baseline

使用 Alpamayo 已有参数：

```bash
python scripts/02_run_baseline_inference.py \
  --run-name alpamayo15_val_candidates_5 \
  --split val \
  --num-traj-samples 5 \
  --execute
```

目标不是立刻改模型，而是先确认多采样候选中是否存在更好的轨迹。

需要比较：

- first-sample baseline：当前默认第一条轨迹。
- oracle best-of-N：用 GT 选择 ADE/FDE 最好的候选，只作为上界。
- random candidate：检查多采样提升不是随机幻觉。

如果 oracle best-of-N 明显优于 first-sample，说明 Alpamayo 的生成分布里有潜力，问题部分来自 candidate selection。

### Step 2: Reasoning-aware reranker

新增一个不使用 GT 的选择器。输入包括：

- Alpamayo 输出的 `cot`、`meta_action`、`answer`。
- 每条 candidate trajectory 的速度、final speed、speed delta、jerk、lateral displacement、heading change。

选择逻辑：

- stop：偏好 final speed 更低、减速更明显、jerk 不爆炸的轨迹。
- yield / slow_down：偏好有减速，但不强制停死。
- turn_left / turn_right：偏好 heading change 方向正确的轨迹。
- avoid / nudge：偏好有合理 lateral displacement 的轨迹。
- 全局 comfort penalty：惩罚 jerk 或 acceleration 过大的轨迹。

输出：

- reranked prediction JSONL。
- reranked trajectories NPZ。
- 每个样本的 score breakdown，解释为什么选择这条轨迹。

### Step 3: 可选的极小速度修正器

如果 reranker 提升有限，但 stop/yield 案例仍明显有可修正空间，可以加一个很保守的 speed-profile adapter。

限制条件：

- 只对高置信 stop / yield / slow_down intent 生效。
- 不改变 lateral path，只调整沿轨迹方向的速度 profile。
- 必须报告 comfort 指标，避免制造急刹或不自然轨迹。
- 必须和 “rerank only” 分开报告，不能混在一起。

这个模块属于 inference-time post-processing，不声称改变 Alpamayo 模型能力。

## 评测设计

核心对比：

| 方法 | 是否用 GT 选轨迹 | 是否训练 | 作用 |
| --- | --- | --- | --- |
| First-sample baseline | 否 | 否 | 当前 Alpamayo 默认输出 |
| Oracle best-of-N | 是 | 否 | 多候选上界，只用于分析 |
| Reasoning-aware reranker | 否 | 否 | 主要方法 |
| Speed-profile adapter | 否 | 否 | 可选后处理 |

必须报告：

- 全 val：ADE、FDE、final speed、jerk。
- stop/yield 子集：ADE、FDE、final speed、speed delta。
- failure cases：reranker 改善了哪些，恶化了哪些。
- oracle gap：reranker 离 best-of-N 还有多远。

成功标准不应该设得夸张。一个合理目标是：

- 在 stop/yield 子集上，final speed 或 FDE 有小幅改善；
- 全 val 不明显恶化；
- 至少 2-3 个 case study 能解释 reranker 为什么选得更好。

## 面试表述

推荐说法：

> 我没有直接微调 10B Alpamayo，因为这在资源和标签条件下不够可信。我的做法是先复现模型并建立可复现评测，然后发现多候选轨迹里存在更合理的行动样本，于是设计了一个 reasoning-aware inference-time reranker，用 CoC intent 和动力学特征选择更合理的轨迹。在不训练大模型的情况下，我对 stop/yield 等长尾场景做了小幅但可解释的改进。

不要说：

- “我修复了 Alpamayo 的 reasoning-action inconsistency。”
- “我让 Alpamayo 学会了更好的驾驶策略。”
- “我接入了世界模型。”
- “我做了 RLHF for driving。”

更稳的说法是：

- “我做了 inference-time trajectory selection。”
- “我把复现扩展成了评测和轻量改进系统。”
- “我用 oracle best-of-N 证明生成候选存在提升空间，再用不依赖 GT 的 reranker 尝试缩小这个 gap。”

## 下一步执行

优先级：

1. 跑 `--num-traj-samples 5` 的 val 候选轨迹。
2. 写 oracle best-of-N evaluator，确认候选上界。
3. 写 reasoning-aware reranker。
4. 复用现有 metrics 比较 first-sample / oracle / reranker。
5. 只对有代表性的改善和恶化案例做可视化。

这条路线正式取代此前的 LoRA / RLHF / consistency repair 主线。
