# 面试稿:Alpamayo-1.5 推理期轨迹选择 + reward-model 探针

> 一页讲稿。主线:**严谨诊断 + reward-model 式探针 + 系统级结论**,不是"我写了个 reranker"。
> 完整数据见 `interview_summary.md` / `PROGRESS.md`。

## 30 秒电梯版

我没有微调 10B 的 Alpamayo,而是研究"它一次采出 5 条候选轨迹、却只部署第一条"这件事。我先用 **oracle 分解**证明**约一半的轨迹误差是可恢复的"选择"误差**;再用一个**免费、无训练的共识(MBR)选择器**显著捞回其中 **~26%**;然后我从**六个角度**(含把模型 hidden state 抽出来训 reward-model 式 verifier)**严谨地证明了打不过这个免费基线**,并给出系统层面的根因:**模型对自己采的样本不暴露任何质量分**。

## 这个项目真正的含金量(主打这三点)

1. **研究判断,不是刷指标。** 我把问题设计成"无论输赢都有干净结论"(误差分解 + faithfulness 探针),而不是赌单一指标上升。
2. **是真 ML,不只是 reranker。** 我往 10B VLA 里挂 forward hook,抽出 VLM 的 hidden state **和扩散 expert 的逐候选激活**,训 reward-model 式 verifier head——这正是现在 LLM test-time scaling 的 **best-of-N + verifier** 范式。
3. **系统洞察 + 严谨统计。** 发现"扩散样本无内部分数"这个根因;全程 bootstrap CI / k-fold / LOO / n=1000 / ablation,负结果也如实报、并能复现。

## 核心数据(官方 val,1000 clips × 5 候选,95% bootstrap CI)

- **误差分解**:first-sample ADE **1.88** → oracle best-of-5 **0.92** ⇒ **~51% 是选择误差**;且 first ≈ random(1.88 vs 1.87)⇒ 选择空间真实、不是运气。
- **免费 MBR/共识选择器**:ADE **1.63**,gap closed **+26%**,CI[0.18, 0.32],**显著**;成本仅 **~1.4×**(不是 5×);overall 与 stop/yield 子集**都显著**。
- **三角 negative(六法皆不胜 MBR)**:
  - reasoning-aware 启发式:不显著,且**不如** reasoning-blind ⇒ CoT 对"选哪条"无预测力(faithfulness 负结果)。
  - learned **ridge**(喂共识特征):**追平** MBR(CI 含 0,776/1000 平局)。
  - learned **分类**(选最优候选):**反噬**(比不选还差,去追离群点)。
  - **set-aggregator**(DeepSets 自学聚合器):赢 first(+11.6%)但**显著输** MBR。
  - **hidden-state verifier**(共享 VLM hidden / 逐候选 expert hidden 的 reward-model 探针):**伤害 / 无信号**。
- **根因(系统层面)**:扩散采样器不返回 likelihood、VLM logprob 只评几乎相同的 CoT ⇒ 模型没有 per-sample 质量分 ⇒ 选择只能靠外部信号,而"样本居中性(MBR)"就是天花板。

## 我声称 / 不声称

- **声称**:量化了可恢复的选择误差;一个免费 MBR 选择器显著捞回 ~26%;并严谨证明 learned / hidden-state reranker 打不过它,且**解释了为什么**。
- **不声称**:改进生成器、微调 10B、世界模型、RLHF、闭环 SOTA。

## 预判三个必问,提前接住

1. **"开环 ADE ≠ 驾驶安全?"** 对,这是离线代理指标;向均值靠拢在多模态场景甚至可能不安全(我有 loss case 说明)。我把项目定位成诊断 + 选择研究,不声称闭环结论。
2. **"MBR 不是新东西?"** 对,MBR / self-consistency 是老想法;我的贡献是*量化它对 Alpamayo 的具体收益* + 三角 negative + "无内部分数"的系统洞察,不是发明 MBR。
3. **"为什么不微调 10B?"** oracle 证明瓶颈在**选择**不在生成;没有质量标签 / 闭环 reward,1000 clip 上微调极易过拟合;而且我的负结果证明瓶颈是**结构性**的(模型构造),不是没调够。

## 收尾(把负结果讲成强项)

> "我用六种方法、从几何特征一路到模型内部激活,严谨地证明了一个免费的两行基线就是天花板,并且我**说得出为什么**——这比报一个我不确定能不能复现的 +2% 更可信。要真突破,只能换信号源(训模型给自己的样本打分 / 闭环或人工 reward),那是另一个更大的项目,我能讲清楚路线。"

## 一句话技能清单(简历/口述用)

复现 10B VLA 推理(流式取数、多候选采样、完整评测 harness)· oracle 误差分解 · 免费 MBR 选择器(显著 +26%)· DeepSets / 逻辑回归 / ridge reranker · VLM & 扩散 expert hidden-state 抽取 + reward-model verifier 探针 · bootstrap CI / k-fold / LOO / 分层抽样 · 诚实的三角 negative + 系统级根因。
