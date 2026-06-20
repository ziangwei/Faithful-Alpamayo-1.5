# Faithful-Alpamayo-1.5 — 实时进度与规划

> 单一事实来源(single source of truth)。每次推进后更新这里。
> 方法论细节见 `technical_route_reassessment_zh.md`;旧的 LoRA/RLHF 路线(`implementation_plan.md`)已废弃。

最后更新:2026-06-20

## 一句话定位

不微调 10B Alpamayo,而是在推理阶段做 **reasoning-aware 轨迹 reranker**:从多候选轨迹里,用 CoC intent + 动力学特征选更合理的一条。核心问题 = 量化"选择误差 vs 生成误差",并检验 reasoning 对选轨迹是否真有用(faithfulness)。

## 当前状态

- ✅ Alpamayo 1.5 单/多 clip 推理跑通(SDPA,单 H100)
- ✅ 数据:官方 PhysicalAI-AV,流式取数(**不占盘**),元数据 parquet 已缓存
- ✅ val 子集构建:官方 val split + 传感器过滤 + 分层抽样(`01b`)
- ✅ 多候选 baseline:60 clip × 5 候选,已落盘(`outputs/runs/val_cand5/`)
- ✅ **Gate B 通过(强绿灯)** —— 见决策记录
- ✅ reranker(aware + blind)写好测好(`04_rerank.py`)
- ⬜ 明日:在 60 clip 上跑 `04_rerank`,看 aware 关掉多少 gap、是否强过 blind
- ⬜ 然后:scale 到 300 clip 出正式数(GPU,~1h)
- ⬜ 之后:置信门控、case study 可视化、(可选)learned reranker

## 决策记录:Gate B(oracle best-of-N,60 clip,2026-06-20)

| 指标 | 全 val (60) | stop/yield (9) |
| --- | --- | --- |
| first-sample ADE | 1.54 m | 2.49 m |
| oracle ADE | 0.66 m | 0.73 m |
| **ADE gap** | **57%** | **71%** |
| FDE gap | 57% | 68% |
| random ≈ first | 是(1.60 vs 1.54) | 是 |
| 可改进 clip 占比 | 78% | 78% |
| 候选 ADE 多样性 std | 0.72 m | 1.06 m |

**结论:绿灯。** 选择误差巨大且真实(random≈first 说明不是运气),候选轨迹多样(尽管 cot 文本几乎一致),stop/yield 子集 gap 更大——正是目标场景。→ 值得做 reranker。

### Tier 1 结果:启发式 reranker(60 clip,2026-06-20)

- 全 val:aware 关掉 gap **-1.8% ADE**(平局偏负),胜负 22:28;**aware ≈ blind**(57/60 选一样)→ intent 几乎没改变选择。
- stop/yield(n=9):aware 关掉 **24% ADE**、20% FDE,5 赢 3 输——目标场景有信号但 n 太薄。
- 判断:**手调启发式弱;质心也不行**——质心 overall 仅 +5.6% 但胜负 25:24(掷硬币),stop/yield **-11.6%(更差)**。即廉价后处理选择 overall 抓不到那 57% gap。
- **唯一亮点:stop/yield 上 aware(+24%)> blind(+16.8%)> first(0)> 质心(-11.6%)**,顺序合理——intent-aware 在安全关键场景有效、且强过 intent-blind(faithfulness 信号),但 n=9。
- 成本已确认:1 候选 ~9s,5 候选 ~13s → 多候选仅 **~1.4×**,非 5×,性价比顾虑排除。
- **决定:scale 到 300**,验证 stop/yield 的 aware>blind 是真信号还是 n=9 噪声(子集→~45)。然后再决定:坐实"reasoning 在 stop/yield 有用"的正面 claim / 转成诊断叙事 / 上 learned reranker。

### n=300 决定性结果(2026-06-20)— 项目主线确定

- **reasoning-aware 启发式被证伪**:overall ADE -0.8%、stop/yield **-12.9%**(n=9 的 +24% 是小样本噪声,scale 后归零转负)。`aware ≈ blind` 处处成立 → intent 不改变选择。**"reasoning 帮选轨迹"的正面 claim 死亡**(诚实负结果;亲手 scale 杀掉自己的假阳性 = 面试强加分)。
- **真正有效:consensus/质心(MBR)选择。** overall ADE gap 关掉 **+16.4%**(胜负 148:89)、FDE ~22%;stop/yield ADE **+18.7%**(29:18)、FDE ~31%。overall 与子集一致为正,n=300 下基本显著。**零训练、零额外推理、有原则。**
- **这是项目正面主线**,框架名 = **Minimum Bayes Risk / self-consistency 选择**(轨迹版,LLM self-consistency 的类比):模型不给 per-轨迹分数,但其 i.i.d. 扩散样本的"共识"稳定打败任意单样本。
- 最终叙事:"Alpamayo 部署的 first-sample 留了 ~50% 可恢复选择误差;MBR/共识选择免费捞回 **16–19%**;手设计的 reasoning-aware 看着 promising(n=9)但 scale 到 300 后归零——一个不被小样本骗的 cautionary tale。"
- 下一步:① 给报告加 bootstrap CI(坐实显著);② 真·medoid 变体(min 两两距离和)对比质心;③(可选)learned reranker 试图打败 MBR;④ 3–5 个 case study + `interview_summary.md`。

## 明日要跑的命令

```bash
# 1) 本地(电脑端)推送
git add scripts/04_rerank.py docs/PROGRESS.md
git commit -m "feat: reasoning-aware/blind reranker + progress doc"
git push origin main

# 2) 服务器:拉取 + 在现有 60 clip 上跑 reranker  [CPU，秒级]
git pull origin main
python scripts/04_rerank.py --run-name val_cand5
```

看 `outputs/runs/val_cand5/analysis/rerank_report.json`:

- `overall.aware_gap_closed_ade_pct` > 0 且 `aware_vs_first` 多赢少输 → reranker 有效。
- `stop_yield_subset.aware_vs_blind` 多赢 → **reasoning 真的帮上忙(faithfulness 证据)**,核心卖点。
- 若 aware ≈ blind → reasoning 没加成;as-is 报告,或调 `score_aware` 权重再试。

确认有效后,**scale 到 300 出正式数**  [GPU，~1h，你有 48h 卡]:

```bash
N=300 RUN=val_cand5_n300 bash scripts/run_baseline.sh       # [GPU]
python scripts/03c_oracle_gap.py --run-name val_cand5_n300   # [CPU]
python scripts/04_rerank.py      --run-name val_cand5_n300   # [CPU]
```

## 管线(脚本链路)

1. `01b_build_val_subset.py` → val 片单(官方 val + 传感器过滤 + 分层)  [CPU]
2. `01_prepare_subset.py` → manifest  [CPU]
3. `02_run_baseline_inference.py --num-traj-samples 5` → 多候选预测+轨迹  [GPU]
4. `03_compute_metrics.py` → 每候选 ADE/FDE + 一致性(可选)  [CPU]
5. `03c_oracle_gap.py` → Gate B:oracle vs first-sample 选择误差  [CPU]
6. `04_rerank.py` → aware/blind reranker + 评测(关掉多少 gap)  [CPU]

一键前三步:`run_baseline.sh`(`N=` 控制 clip 数,`LIMIT=` 做冒烟测试)。

## 关键事实备忘(踩过的坑)

- **流式不占盘**:推理按 clip 流式取图(4 相机 × 4 帧,1080×1920),不写硬盘;每次跑重新拉(网络),单 clip 解码后几百 MB 内存瞬时。所以 300 甚至全量都不占配额。
- **Xet 必须禁用**:`export HF_HUB_DISABLE_XET=1`,否则下载/流式得到 0 字节(集群连不上 Xet 端点)。已写死在 `run_baseline.sh`。
- **HF token 坑**:别设 `HF_HOME`(会把 token 路径指到错地方 → 401);要控制下载位置用 `--cache-dir` 或 `HF_HUB_CACHE`,token 留默认。
- **数据没有 `vla_golden.parquet`**:索引是 `clip_index.parquet`(clip_is_valid/chunk/split);元数据 `metadata/data_collection.parquet`(country/month/hour_of_day/platform_class/radar_config)+ `feature_presence.parquet`(各传感器存在标志)。三表等长同序,**clip_id 是元数据表的索引**(UUID)。
- **模型**:`nvidia/Alpamayo-1.5-10B`,bf16,`attn_implementation=sdpa`(flash-attn 编不了),单 H100 够。
- **官方 split**:train 153625 / val 90928 / test 61599。

## 路线图(Tier)与当前位置

- **Tier 0(必交)**:✅ oracle-gap 分解 + 失效分析骨架 + 数据/管线。← 已具备
- **Tier 1(进行中)**:reasoning-aware vs blind reranker + 评测。← **明日推进**
- **Tier 2(有时间)**:极小 learned reranker(逻辑回归,<10 特征,leave-one-clip-out CV);置信门控"介入率 vs 改善"曲线。
- **Tier 3(按需)**:保守 speed-profile adapter(仅高置信 stop/yield,只调沿轨迹速度 profile,不改 lateral)。

## 面试 caveats(主动说,体现严谨)

- oracle 用 GT,是**上界**;reranker 抓不到全部,但 57% 头顶空间意味着抓一部分就有意义。
- n 要够:60 用于开发,**正式数用 300**,stop/yield 子集才到 ~45,结论才可信。
- 报 per-case 输赢 + bootstrap CI + 效应量,不吹"显著提升"。
- 多样性在轨迹动力学、不在 cot 文本——所以 reranker 用动力学特征是对的;intent 当 per-clip 先验。
- 不声称"修复了 Alpamayo / 接了世界模型 / 做了 RLHF"。说法:"inference-time trajectory selection + 误差分解 + 轻量改进系统"。

## 待办 / 未来

- [ ] 明日:`04_rerank` on 60 → 判定 aware vs blind、关掉多少 gap。
- [ ] scale 300:baseline + oracle + rerank,出正式表。
- [ ] 给 `03c` / `04` 的聚合加 bootstrap CI。
- [ ] 3–5 个 case study 可视化(相机帧 + cot + first vs reranked 轨迹 + 速度曲线);用 `rerank_selection.jsonl` 挑改善/恶化案例。
- [ ] (可选)置信门控 `--gate-margin` 扫描,画介入率 vs 改善。
- [ ] 写 `docs/interview_summary.md`。
