# Faithful-Alpamayo-1.5 — 项目状态与交接(单一事实来源)

> **新会话从这里读起。** 本文件 + `docs/interview_summary.md` 足以了解全局并直接开工 B。
> 方法细节见 `technical_route_reassessment_zh.md`;旧 LoRA 路线 `implementation_plan.md` 已废弃。

最后更新:2026-06-20

## 1. 一句话 + 现在的结论

不微调 10B Alpamayo,在推理阶段从它生成的多条候选轨迹里**选**更好的一条。

**结论(n=1000 已确证)**:模型部署的 first-sample 留了 **~51% 可恢复的*选择*误差**;一个免费、无训练的 **consensus / Minimum-Bayes-Risk 选择器**显著捞回 **~26% ADE / ~26% FDE**(overall 与 stop/yield 子集**均显著**);而 **reasoning-aware 选择无效、且不如 reasoning-blind**(intent 对选轨迹无信号)。

**B 结论(learned reranker,n=1000 已确证)**:放开手训练也**打不过**免费 MBR。ridge 回归(预测 ADE 选 argmin)**追平** MBR——ADE 1.634 vs 1.635,+25.8% vs +25.7%,vs-MBR CI[−0.018, 0.019] 含 0、776/1000 平局——坐实"MBR 是廉价后验特征的天花板"。把目标设成"分类最优候选"(logreg)反而**比不选还差**(−5.4%,输 MBR CI[−0.39, −0.21]):oracle-best 多是偏离共识的走运抽样,分类器学会追离群点 → 平均反噬。**回归学到"居中更安全",分类学到"追离群"——目标设定决定成败。**

## 2. 关键数据结果(官方 val,1000 clips × 5 候选,95% bootstrap CI)

误差分解(oracle 仅用于分析,不用于选择):
- first-sample ADE **1.880** / FDE **5.693**;oracle best-of-5 ADE **0.924** / FDE **2.839** → **~51% 是选择误差**。
- first ≈ random(1.880 vs 1.865)→ 选择空间真实,非运气。

选择器对比(均不用 GT):

| 方法 | ADE | FDE | vs first(ADE 改善, 95%CI) |
| --- | --- | --- | --- |
| first-sample(部署) | 1.880 | 5.693 | — |
| reasoning-aware | 1.815 | 5.605 | +0.07, CI[−0.02, 0.16] **不显著** |
| reasoning-blind | 1.772 | 5.445 | +11%(粗糙"居中"启发式) |
| **consensus / MBR** | **1.635** | **4.964** | **+0.25, CI[0.18, 0.32]**, frac=1.0 |
| oracle best-of-5 | 0.924 | 2.839 | 上界 |

- **consensus 强显著**:ADE gap closed **+25.7%**,CI[0.18, 0.32];FDE +0.73 CI[0.49, 0.97];胜负 vs first = **498/305/197**。
- **stop/yield 子集(n=212)也显著**:ADE +0.27 CI[0.12, 0.42] frac=1.0;FDE +0.82 CI[0.37, 1.34];gap closed 27%。
- **reasoning 不 faithful**:reasoning-aware(+7%)**不显著、且不如 reasoning-blind(+11%)**(头对头 60 赢 80 输;stop/yield FDE 显著为负 frac 0.056)。intent 条件化对选轨迹无信号、略有害;有用的信号是纯"居中/共识"。
- 机制(case study):consensus 赢在救回离谱的 first-sample(4.50→0.79),输在 first 本就极好时被拉向均值(0.46→4.21)。
- 成本:多候选仅 **~1.4×**(1 候选 ~9s,5 候选 ~13s),非 5×。

## 3. 数据 & 管线

- 数据集 `nvidia/PhysicalAI-Autonomous-Vehicles` 官方 val。**流式取数,不占盘/不占 inode**。
- **没有 `vla_golden.parquet`**;索引是 `clip_index.parquet`(`clip_is_valid`/`chunk`/`split`),元数据 `metadata/data_collection.parquet`(`country`/`month`/`hour_of_day`/`platform_class`/`radar_config`,**clip_id 是索引**)+ `feature_presence.parquet`(各传感器存在标志)。三表等长同序。
- 模型 `nvidia/Alpamayo-1.5-10B`,bf16,`attn_implementation=sdpa`,单 H100。轨迹由**官方 FlowMatching 扩散器**采样,**不返回 per-轨迹分数**(关键:所以必须外部选择)。

脚本链路:

1. `01b_build_val_subset.py` — 官方 val + 传感器过滤 + 分层抽样 → 片单 [CPU]
2. `01_prepare_subset.py` — → manifest [CPU]
3. `02_run_baseline_inference.py --num-traj-samples 5` — 多候选预测+轨迹 [GPU]
4. `03c_oracle_gap.py` — oracle gap 分解 [CPU]
5. `04_rerank.py` — aware/blind/consensus 选择器 + bootstrap CI [CPU]
6. `05_case_studies.py` — win/loss 案例 + 双栏图(top-down 轨迹 + 速度曲线 + CoT 注释)[CPU]
7. `06_learned_rerank.py` — **B**:learned reranker(ridge + 逻辑回归,leave-one-clip-out CV)+ vs-first / vs-MBR bootstrap [CPU,numpy-only,无需 GPU]
8. `07_set_rerank.py` — **C**:set-aggregator reranker(numpy DeepSets,permutation-invariant,k-fold CV)+ vs-MBR bootstrap [CPU,无需 GPU]
9. `08_train_verifier.py` — **2.0**:hidden-state verifier head(numpy MLP,**geom vs geom+scene 消融**,k-fold CV)+ vs-MBR bootstrap [CPU;先 `02 --dump-hidden`(GPU)产出 scene 向量]

一键 1–3:`run_baseline.sh`(`N=` 控制 clip 数,`RUN=` 命名,`LIMIT=` 冒烟)。最终 run:`N=1000 RUN=val_cand5_n1000`。

**必知的坑:**
- `export HF_HUB_DISABLE_XET=1` 必须设,否则下载/流式得 0 字节(集群连不上 Xet)。已写死在 `run_baseline.sh`。
- 别设 `HF_HOME`(token 会被指到错路径 → 401);下载位置用 `--cache-dir`,token 留默认登录处。
- 现有产出在 `outputs/runs/val_cand5_n1000/`(**gitignore,不进仓库**):`baseline/`(预测+轨迹+GT)、`analysis/`(oracle_gap / rerank_report / rerank_selection / case_studies)、`figures/`。

## 4. 已完成 / 当前位置

- ✅ A 全部完成(n=1000 定稿):复现、数据、管线、oracle 诊断、consensus 显著结果、reasoning 负结果、bootstrap CI、case study、`docs/interview_summary.md`。
- ✅ **B 完成(n=1000 真实数字已出)**:`scripts/06_learned_rerank.py`(numpy-only,LOO-CV)+ `tests/test_learned_rerank.py`(6 测试)。**结果:ridge 追平 MBR(+25.8%,vs-MBR CI 含 0、776/1000 平局),logreg 比不选更差(−5.4%)。结论 = learned 打不过免费 MBR,MBR 是廉价特征天花板;目标设定(回归 vs 分类)本身是干净的方法学结论。**
- ✅ **C(set-aggregator)完成(n=1000 真实数字已出)**:`scripts/07_set_rerank.py`(numpy DeepSets,梯度检验)+ `tests/test_set_rerank.py`(5 测试)。**结果:set-net 显著赢 first(+11.6% gap,CI[0.028,0.199])但显著输 MBR(−0.135,CI[−0.199,−0.071]),只捞回 MBR 一半的 gap;stop/yield 子集与 MBR 打平(CI 含 0)。故意不喂 dist_to_consensus、逼它自学聚合器 → 学出比算术均值更糙的共识。**
- 🔑 **三重印证(纯几何选择见顶)**:B-ridge(线性,*喂*共识特征)**追平** MBR;B-logreg(分类)**反噬**;C-set-net(非线性,*自学*聚合)**显著输**。三种独立学习方法都打不过免费均值共识 → **几何选择在 MBR 见顶**,唯一杠杆是模型内部场景表征。
- 🚀 **2.0 = hidden-state verifier head**:设计见 `docs/tier3_hidden_state_verifier.md`(v2.0);head 训练/评测脚本 `scripts/08_train_verifier.py`(+ 测试)已就绪,核心是 **geom-only vs geom+scene 消融**,直接量化"内部状态比纯几何多带多少选择信号"。待服务器 dump 一次 hidden state(需 GPU)后训 head。`05_case_studies.py` 已升级双栏图(轨迹+速度+CoT)。

## 5. B 交接 brief:learned reranker(Tier 2)

> **状态(2026-06-20):完成,n=1000 真实数字已出。** `scripts/06_learned_rerank.py`(numpy-only,无 sklearn)+ `tests/test_learned_rerank.py`(6 测试)。运行:`python scripts/06_learned_rerank.py --run-name val_cand5_n1000`。
> **结果(overall):ridge ADE 1.634 ≈ MBR 1.635(+25.8% vs +25.7%,vs-MBR CI[−0.018, 0.019] 含 0、776/1000 平局)→ 追平、打不过;logreg ADE 1.932 > first 1.880(−5.4%,vs-MBR CI[−0.39, −0.21] frac 0)→ 显著更差。** stop/yield 同形:ridge +27.7% ≈ MBR +27.1%,logreg +10.6% 更差。
> 机制:回归 ADE → 学到"居中更安全"、收敛到 MBR;分类 is_best → oracle-best 多为偏离共识的走运抽样 → 学会追离群点 → 反噬。**learned 打不过免费 MBR,坐实廉价特征天花板;想真超过得上 richer(视觉/hidden-state)特征。**

**核心问题:** 一个学出来的小评分器,能不能打败**免费的 MBR/consensus**?

**用什么数据:** 现有 `outputs/runs/val_cand5_n1000/baseline/`(1000 clip × 5 候选,预测+轨迹+GT)。**不需要再跑 GPU**——用 leave-one-clip-out CV 即可。

**做法:**
- 候选特征:`04_rerank.py::candidate_features`(末速 / speed_delta / jerk / heading / lateral)+ **到共识的距离**(已知有信号)。
- 标签:该候选是否 ADE 最低(分类)或回归 ADE。
- LOO-CV:每个 clip 用其余 999 训一个小模型,预测这 5 条的分,选 argmax,评 ADE。**已实现两种(numpy 手写,仓库无 sklearn):ridge 回归(预测 clip 内 z-scored ADE,选 argmin)+ L2 逻辑回归(预测 is_best,选 argmax)。**
- 对比 first / **MBR** / ridge / logreg / oracle,带 paired bootstrap CI:vs-first(同 A 的方法)**和 vs-MBR(关键对照)**,overall + stop/yield 子集,均带 per-case 输赢。

**诚实预期(务必先想清楚):** 已证明廉价动力学 / intent 特征**不带选择信号**(aware 不如 blind)。所以 learned-over-cheap-features **很可能 ≈ MBR、打不过**——这本身是干净结论("MBR 已接近廉价特征选择的天花板")。**想真正打败 MBR,需要更 richer 的特征**(视觉 / 场景 embedding,或模型 hidden states),那是更大的工程。新会话开工前先决定:接受"learned≈MBR"的确认性结论,还是投入 richer 特征去搏一把。

**关键文件:** `scripts/04_rerank.py`(`candidate_features` / 质心逻辑 / `bootstrap_improvement`)、`faithful_vla/metrics.py`(`compute_trajectory_metrics` / `parse_intents`)、`faithful_vla/run_paths.py`(输出路径)。

## 6. 面试叙事(详见 `interview_summary.md`)

- 正面:"Alpamayo 部署的 first-sample 留 ~51% 可恢复选择误差;免费 MBR/共识选择显著捞回 ~26%,因为扩散样本无内部分数、有用信号是样本居中性。"
- 负面(加分项):reasoning-aware 在 n=9 看着 +24%,scale 到 1000 后不显著且不如 reasoning-blind——靠 scale 亲手杀掉假阳性。
- 不声称:改生成 / 微调 10B / 世界模型 / RLHF。

## 7. 简要决策日志

- 否定 LoRA/RLHF/重型世界模型:资源不够 + 靶子不对(oracle 证明瓶颈是**选择**,不是生成;10B 生成已够好)。
- Gate B(oracle gap)绿灯:选择误差 57%(n=60)/ **51%(n=1000)**。
- reranker:手调启发式 n=60 平局 → 加 consensus → **n=1000:consensus 强显著(overall + stop/yield)、aware 不显著且不如 blind**;n=9 的 stop/yield 正信号被 scale 证伪。

## 8. 样本量:已定稿在 n=1000

- 1000 个 held-out clip + bootstrap CI,"too few" 的质疑已彻底排除(AV 评测通常几百到几千,且每个还是一次 10B 推理)。
- overall consensus(ADE/FDE)与 **stop/yield 子集(n=212)均显著**(CI 不含 0,frac=1.0)。
- 对 B:1000 用 LOO-CV 足够下结论,也够训简单模型。若 learned 与 MBR 差距极小仍可能分不清——但那本身说明"没差距"。不必再加跑。
