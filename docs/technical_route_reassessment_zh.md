# 技术路线重评 v2（执行版）：否定旧路线，转向最小可验证改进

日期：2026-06-18
版本说明：v2 在原重评基础上并入优化项，按 Tier 分层并加入 go/no-go 闸，便于评审与排期。

本文档覆盖此前的技术路线。结论不变：不要把主线放在“言行不一致修复”、LoRA 微调或 RLHF 上；它们作为背景和辅助分析保留，不作为核心贡献。v2 进一步把项目重心从“刷一个指标”转向一个无论实验输赢都有结论的研究问题。

## 0. 一句话定位

本项目要回答三件事，而不是赌单一指标上升：

1. **误差分解**：Alpamayo 的轨迹误差里，多少是“选择误差”（更好的候选已被采样出来，只是没被选中），多少是“生成误差”（更好的候选根本没被采出来）。
2. **逼近上界**：用一个不训练 10B 的 reasoning-aware inference-time reranker，尝试把“选择误差”那部分捞回来。
3. **faithfulness 探针**：Alpamayo 的 CoC reasoning 对“选哪条轨迹更好”到底有没有预测力 —— 正好呼应项目名 Faithful。

其中 (1) 和 (3) 无论 reranker 成败都有干净结论，是本路线的抗风险底座。

## 1. 当前事实

已完成 Alpamayo 1.5 的基础复现链路：

- 能在 PhysicalAI-AV 子集上跑 baseline inference。
- 已建立 named run 输出目录，避免实验结果混乱。
- 已有 ADE/FDE、速度、加速度、jerk、reasoning-action consistency 评测脚本。
- 已有 failure mining report，可定位 stop/yield 等高风险样本。
- baseline 脚本已支持 `--num-traj-samples` / `--num-traj-sets`，且每条候选的 `pred_xyz`、`cot`、`meta_action`、`answer` 与 GT 轨迹均已落盘 —— 多候选 + oracle 分析无需改核心代码。

价值在于工程能力与研究判断；但若只停在“复现 + 测评 + 失效分析”，缺少“我让系统变好了一点”的变化。因此下一步加一个很小但可验证的改进模块。

## 2. 明确否定的旧路线

以下四条仍然否定，保留判断过程供评审参考。

### 2.1 不以“言行不一致修复”为主线

指标依赖脆弱的文本 intent 解析，阈值主观，易 false positive；只优化它容易变成刷自定义 metric，不代表驾驶行为真的更好。保留 reasoning-action consistency 作为 diagnostic 辅助指标，并在报告中诚实说明其误判，需结合轨迹、速度曲线和视觉证据解释。

### 2.2 不以 LoRA / SFT 微调 Alpamayo 为主线

10B 级 VLA 训练成本高；当前无高质量人工修正标签、无稳定任务级 reward；60 val clips 上极易过拟合，提升无法与采样、随机性、阈值变化区分。暂不训练大模型；若后续必须有学习模块，只做轻量 reranker/calibrator，不动 Alpamayo 本体。

### 2.3 不以 RLHF / 强化学习为主线

官方 RL post-training 需多卡大显存；缺真实闭环或可靠 pseudo-closed-loop reward 时结论难站住；reward hacking 风险高。仅在 related/future work 提及，当前不实现。

### 2.4 不以重型世界模型接入为主线

“CoT 条件世界模型生成未来 BEV，再反喂 Alpamayo 修正决策”是完整研究项目，工程面过大。仅保留 world-model-inspired 思想：用未来轨迹、速度曲线和场景证据做 failure audit；不声称训练或接入完整世界模型。

## 3. 新主线：Reasoning-Aware Trajectory Reranker

核心不是训练 Alpamayo，而是在推理阶段加一个轻量模块：

> Alpamayo 生成多个候选轨迹；一个小模块根据 CoC/intent 和轨迹动力学特征，选择（必要时轻微修正）更合理的轨迹。

好处：不训练 10B、不下新大数据、不依赖闭环仿真、与现有 baseline / metrics / failure mining 直接衔接，并能给出清晰对比 first-sample vs oracle best-of-N vs reranker。

### 3.1 两条对照，锁定 faithfulness 这个真问题

- **reasoning-aware reranker**：用 CoC/intent + 动力学特征选择。
- **reasoning-blind reranker**：去掉 cot/intent，只用动力学特征（final speed、speed delta、jerk、heading change、lateral displacement）。

若 aware 明显优于 blind → CoC 对动作有预测力（faithful）的证据；若无差 → reasoning 部分是装饰性的。两种结果都构成结论。这个 ablation 是科学主线，不是附属，也避免“reranker 靠的是脆弱 intent 解析”的质疑（intent 此处只当选择先验，选错只是少捞一点，不是刷假指标）。

### 3.2 置信度门控的选择性干预

reranker 不无脑覆盖第一条候选，只在高置信认为替代更优时介入，否则沿用 Alpamayo 默认输出。

- “全 val 不恶化”几乎由构造保证：介入率旋钮可直接守住成功标准里最易翻车的那条。
- 可画“介入率 vs 改善”曲线，表达部署 / 安全权衡，是很 AV 的表述。

## 4. 分层交付与 go/no-go 闸

执行用两道闸控制，避免在没有上升空间时空跑 reranker：

- **Gate A —— 候选多样性诊断**：量 N 条候选的两两轨迹距离与 intent 一致性。多样性过低 → 先提 temperature / 加 `num_traj_sets`；仍低 → 不做 reranker，退回 Tier 0。
- **Gate B —— oracle gap**：oracle best-of-N 是否明显优于 first-sample。gap 太小 → 选择误差不是瓶颈，贡献收回到“误差分解 + 失效分析”（仍是完整结论）。

交付分层，确保任何阶段都有可讲的东西：

| Tier | 内容 | 触发条件 | 风险定位 |
| --- | --- | --- | --- |
| Tier 0（必交） | 候选多样性诊断 + oracle-gap 分解 + 失效 taxonomy + 3–5 个可视化 case | 无条件 | 哪怕 reranker 全败，已是完整项目 |
| Tier 1（大概率） | reasoning-aware vs reasoning-blind 启发式 reranker + 置信度门控 | 过 Gate A、B | 主方法 + 核心 ablation |
| Tier 2（有时间） | 极小 learned reranker（逻辑回归 / 深度 2 树，<10 特征，leave-one-clip-out CV，如实报 train/val gap） | Tier 1 有信号 | 给“学习”成分，不碰 10B、不冒过拟合大险 |
| Tier 3（按需） | 极保守 speed-profile adapter | Step 2 后 stop/yield 仍有明显 gap | 风险最高，单独汇报 |

## 5. 具体步骤

### Step 0：候选多样性诊断（Gate A）

读已保存的多候选输出，算两两轨迹距离、intent 一致率。一小时级别即可判断方向是否有希望。这是比 oracle gap 更早的金丝雀。

### Step 1：多候选 baseline + oracle（Gate B）

复用现有脚本：

```bash
python scripts/02_run_baseline_inference.py \
  --run-name alpamayo15_val_candidates_5 \
  --split val \
  --num-traj-samples 5 \
  --execute
```

对比：

- first-sample baseline：当前默认第一条轨迹。
- oracle best-of-N：用 GT 选 ADE/FDE 最优的候选，仅作上界。
- random candidate：检查多采样提升不是随机幻觉。

若 oracle best-of-N 明显优于 first-sample，说明 Alpamayo 的生成分布里有潜力，问题部分来自 candidate selection。

### Step 2：reasoning-aware + reasoning-blind reranker（含门控）

不使用 GT 的选择器。输入：每条候选的 `cot` / `meta_action` / `answer` 与动力学特征。

aware 版选择逻辑：

- stop：偏好 final speed 更低、减速明显、jerk 不爆炸。
- yield / slow_down：偏好有减速，但不强制停死。
- turn_left / turn_right：偏好 heading change 方向正确。
- avoid / nudge：偏好合理 lateral displacement。
- 全局 comfort penalty：惩罚 jerk / acceleration 过大。

blind 版去掉 cot/intent，只用动力学特征，作为 faithfulness 对照。

输出：reranked prediction JSONL、reranked trajectories NPZ、每样本 score breakdown（解释为何选它）、介入与否标记。

### Step 3（可选，Tier 2）：极小 learned reranker

在 train split 候选特征上训一个轻量分类 / 排序器，预测哪条候选 ADE 最低。<10 特征、强正则、leave-one-clip-out CV，如实报 train/val gap。不触碰 Alpamayo，纯 meta-model。

### Step 4（可选，Tier 3）：极保守 speed-profile adapter

仅对高置信 stop / yield / slow_down 生效；只调沿轨迹方向的速度 profile，不改 lateral path；必报 comfort，避免制造急刹；与 rerank-only 分开汇报。属 inference-time 后处理，不声称改变模型能力。

## 6. 评测设计

核心对比：

| 方法 | 用 GT 选轨迹 | 训练 | 作用 |
| --- | --- | --- | --- |
| First-sample baseline | 否 | 否 | 当前 Alpamayo 默认输出 |
| Oracle best-of-N | 是 | 否 | 多候选上界，仅用于分析 |
| Random candidate | 否 | 否 | 防随机幻觉对照 |
| Reasoning-blind reranker | 否 | 否 | faithfulness 对照 |
| Reasoning-aware reranker | 否 | 否 | 主方法 |
| + 置信度门控 | 否 | 否 | 控制介入率，守不恶化 |
| Learned reranker | 否 | 是（CPU 小模型） | 可选上限探测 |
| Speed-profile adapter | 否 | 否 | 可选后处理 |

必须报告：

- 全 val：ADE、FDE、final speed、jerk。
- stop/yield 子集：ADE、FDE、final speed、speed delta。
- failure cases：reranker 改善了哪些，恶化了哪些。
- oracle gap：reranker 离 best-of-N 还有多远。

统计克制（把小 n 的弱点转成严谨度信号）：

- n 透明：明确 val 和 stop/yield 子集样本数。
- paired bootstrap CI + per-case 输赢 / 平表，不只报均值。
- 不在 60 clips 上声称显著 SOTA。

成本：报多采样的 latency（N× 推理）与部署权衡。

合理成功标准（不夸张）：

- stop/yield 子集上 final speed 或 FDE 有小幅改善；
- 全 val 不明显恶化；
- 2–3 个 case study 能解释 reranker 为什么选得更好。

## 7. 可视化与 demo 抓手

面试真正记得住的是图，不是表。准备 3–5 个 before/after：相机帧 + CoC reasoning 文本 + first-sample 轨迹 vs reranked 轨迹 + 速度曲线，并附 reranker 的一句话“为什么选它”。典型范例：reasoning 说为行人停车，默认轨迹却滚过斑马线，reranker 选中真正减速那条。

## 8. 面试表述

推荐说法：

> 我没有直接微调 10B Alpamayo，因为资源和标签条件下不够可信。我先复现模型并建立可复现评测，再用 oracle best-of-N 量化出轨迹误差里有多少是可恢复的“选择误差”，然后设计了一个不依赖 GT、不训练大模型的 reasoning-aware inference-time reranker 去逼近这个上界，并用 reasoning-blind 对照检验 CoC 是否真的对动作有预测力。在 stop/yield 等长尾场景上做了小幅但可解释的改进。

不要说：

- “我修复了 Alpamayo 的 reasoning-action inconsistency。”
- “我让 Alpamayo 学会了更好的驾驶策略。”
- “我接入了世界模型。”
- “我做了 RLHF for driving。”

更稳的说法：

- “我做了 inference-time trajectory selection。”
- “我把复现扩展成了评测 + 误差分解 + 轻量改进系统。”
- “我用 oracle best-of-N 证明候选里有提升空间，再用不依赖 GT 的 reranker 尝试缩小这个 gap。”

## 9. 执行优先级（按闸推进）

1. Step 0 多样性诊断 → 过 Gate A。
2. Step 1 多候选 + oracle evaluator → 过 Gate B（全项目成立与否系于此）。
3. Step 2 aware + blind reranker + 门控，复用现有 metrics 比较。
4. 仅对代表性改善 / 恶化案例做可视化（Tier 0 交付物）。
5. 有余力再上 Tier 2 learned reranker；stop/yield 仍有 gap 才上 Tier 3 adapter。

本路线正式取代此前 LoRA / RLHF / consistency repair 主线。
