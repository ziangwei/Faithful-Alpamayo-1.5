# Faithful-Alpamayo-1.5 — 项目状态与交接(单一事实来源)

> **新会话从这里读起。** 本文件 + `docs/interview_summary.md` 足以了解全局并直接开工 B。
> 方法细节见 `technical_route_reassessment_zh.md`;旧 LoRA 路线 `implementation_plan.md` 已废弃。

最后更新:2026-06-20

## 1. 一句话 + 现在的结论

不微调 10B Alpamayo,在推理阶段从它生成的多条候选轨迹里**选**更好的一条。

**结论(n=300 已确证)**:模型部署的 first-sample 留了 **~47% 可恢复的*选择*误差**;一个免费、无训练的 **consensus / Minimum-Bayes-Risk 选择器**显著捞回 **~16% ADE / ~22% FDE**;而 **reasoning-aware 选择无效(null)**。

## 2. 关键数据结果(官方 val,300 clips × 5 候选,95% bootstrap CI)

误差分解(oracle 仅用于分析,不用于选择):
- first-sample ADE **1.762** / FDE **5.524**;oracle best-of-5 ADE **0.925** / FDE **2.949** → **~47% 是选择误差**。
- first ≈ random(1.762 vs 1.851)→ 选择空间真实,非运气。

选择器对比(均不用 GT):

| 方法 | ADE | FDE | vs first(ADE 改善, 95%CI) |
| --- | --- | --- | --- |
| first-sample(部署) | 1.762 | 5.524 | — |
| reasoning-blind | 1.805 | 5.586 | 负,不显著 |
| reasoning-aware | 1.769 | 5.578 | **null**: +0.00, CI[−0.15, 0.15] |
| **consensus / MBR** | **1.625** | **4.958** | **+0.14, CI[0.03, 0.25]**, 99%>0 |
| oracle best-of-5 | 0.925 | 2.949 | 上界 |

- **consensus 显著**:ADE CI 不含 0;FDE +0.57 CI[0.18, 0.96];胜负 vs first = **148/89/63**。
- **reasoning-aware ≈ reasoning-blind(260/300 tie)** → CoC reasoning 对选轨迹**无额外信号**(faithfulness-via-reranking 失败,诚实负结果)。
- **stop/yield 子集(n=61)**:consensus FDE 显著 CI[0.05, 1.63],**ADE 仅 suggestive**(+0.15, 89%>0, CI 跨 0)→ 子集 n 不够(见 §8)。
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
6. `05_case_studies.py` — win/loss 案例 + 轨迹图 [CPU]

一键 1–3:`run_baseline.sh`(`N=` 控制 clip 数,`RUN=` 命名,`LIMIT=` 冒烟)。

**必知的坑:**
- `export HF_HUB_DISABLE_XET=1` 必须设,否则下载/流式得 0 字节(集群连不上 Xet)。已写死在 `run_baseline.sh`。
- 别设 `HF_HOME`(token 会被指到错路径 → 401);下载位置用 `--cache-dir`,token 留默认登录处。
- 现有产出在 `outputs/runs/val_cand5_n300/`(**gitignore,不进仓库**):`baseline/`(预测+轨迹+GT)、`analysis/`(oracle_gap / rerank_report / rerank_selection / case_studies)、`figures/`。60-clip 旧 run 在 `outputs/runs/val_cand5/`。

## 4. 已完成 / 当前位置

- ✅ A 全部完成:复现、数据、管线、oracle 诊断、consensus 显著结果、reasoning-aware 负结果、bootstrap CI、case study、`docs/interview_summary.md`。
- ⬜ **B = learned reranker(下一步,新会话主题)。**

## 5. B 交接 brief:learned reranker(Tier 2)

**核心问题:** 一个学出来的小评分器,能不能打败**免费的 MBR/consensus**?

**用什么数据:** 现有 `outputs/runs/val_cand5_n300/baseline/`(300 clip × 5 候选,预测+轨迹+GT)。**不需要再跑 GPU**——用 leave-one-clip-out CV 即可。

**做法:**
- 候选特征:`04_rerank.py::candidate_features`(末速 / speed_delta / jerk / heading / lateral)+ **到共识的距离**(已知有信号)。
- 标签:该候选是否 ADE 最低(分类)或回归 ADE。
- LOO-CV:每个 clip 用其余 299 训一个小模型(逻辑回归 / GBM),预测这 5 条的分,选 argmax,评 ADE。
- 对比 first / **MBR** / oracle,带 bootstrap CI + vs-MBR 输赢(直接复用 `04` 的 `bootstrap_improvement`)。

**诚实预期(务必先想清楚):** 已证明廉价动力学 / intent 特征**不带选择信号**(aware/blind 全 null)。所以 learned-over-cheap-features **很可能 ≈ MBR、打不过**——这本身是干净结论("MBR 已接近廉价特征选择的天花板")。**想真正打败 MBR,需要更 richer 的特征**(视觉 / 场景 embedding,或模型 hidden states),那是更大的工程。新会话开工前先决定:接受"learned≈MBR"的确认性结论,还是投入 richer 特征去搏一把。

**关键文件:** `scripts/04_rerank.py`(`candidate_features` / 质心逻辑 / `bootstrap_improvement`)、`faithful_vla/metrics.py`(`compute_trajectory_metrics` / `parse_intents`)、`faithful_vla/run_paths.py`(输出路径)。

## 6. 面试叙事(详见 `interview_summary.md`)

- 正面:"Alpamayo 部署的 first-sample 留 ~47% 可恢复选择误差;免费 MBR/共识选择显著捞回 ~16%/22%,因为扩散样本无内部分数。"
- 负面(加分项):reasoning-aware 在 n=9 看着 +24%,scale 到 300 后归零/转负——靠 scale 亲手杀掉假阳性。
- 不声称:改生成 / 微调 10B / 世界模型 / RLHF。

## 7. 简要决策日志

- 否定 LoRA/RLHF/重型世界模型:资源不够 + 靶子不对(oracle 证明瓶颈是**选择**,不是生成;10B 生成已够好)。
- Gate B(oracle gap)绿灯:选择误差 57%(n=60)/ 47%(n=300)。
- reranker:手调启发式 n=60 平局 → 加 consensus → n=300 **consensus 显著、aware null**;n=9 的 stop/yield 正信号被 n=61 证伪。

## 8. "300 够不够下结论?"(我的判断)

- **总体 consensus 结论:够。** n=300 下 CI 不含 0(ADE/FDE 都显著)。
- **reasoning-aware null:够。** CI 紧贴 0。
- **stop/yield 子集:不够。** n=61,ADE CI 跨 0。若要把子集 ADE 做成显著结论,需总量 ~800–1500(子集→~150–300)。成本低(~1.4×、流式、不占盘),48h 卡内可行,但**非必需**。
- **B(learned reranker,LOO-CV):300 够做首轮结论**,也够训一个简单模型;但若 learned 与 MBR 差距很小,300 难分辨,且 300 对复杂模型偏薄。要更稳就 scale 到 ~600–1000。
