# 2.0 设计:frozen-VLM hidden-state 上的 learned verifier head

日期:2026-06-20
状态:**v2.0 已跑出首个结果(见 §9):候选共享的 scene 反而显著伤害。** 已加两手对策:`08 --scene-pca`(降维治过拟合)+ `02 --dump-expert-hidden`(逐候选 expert hidden,绕开 shared 限制)。测试:`test_train_verifier.py`(6)+ `test_dump_hidden.py`(8)全过。不微调 10B。

## 1. 为什么(1.0 的三重印证)

1.0 已经从三个角度证明**纯几何的选择空间在 MBR 见顶**:

| 方法 | 给什么 | 结果 vs MBR |
| --- | --- | --- |
| B-ridge(线性回归) | *喂* dist-to-consensus | **追平**(CI 含 0、776/1000 平局) |
| B-logreg(分类最优候选) | 同上 | **反噬**(比不选还差 −5.4%) |
| C-set-net(非线性,DeepSets) | 让它*自学*聚合器 | **显著输**(只捞回 MBR 一半 gap) |

三种独立学习方法都打不过免费的均值共识。结论:瓶颈不是模型容量,是**信息量**——候选轨迹的 (x,y) 几何里没有"为什么这条更安全"的场景依据。模型内部其实知道(前视相机里有没有行人/红灯/要不要让行),但采样出 (x,y) 后这些就丢了。**要破天花板,只能把模型的内部状态接出来当特征。**

## 2. 核心思想:best-of-N + reward model

> 冻结 Alpamayo;对每条候选,从 VLM 内部状态取一个池化向量;训一个 <100K 参数的 verifier head 预测候选质量(ADE 排序),推理时选 argmin。

这正是现代 LLM test-time scaling 的标准范式(采样 N 条 + 一个学出来的 verifier/PRM 打分),搬到驾驶 VLA。不动 10B 主体,只训一个小 head;评测口径与 A/B/C 完全一致(k-fold/LOO CV + paired bootstrap vs MBR + stop/yield 子集)。

## 3. 锋利的实验:geom vs geom+scene 消融(`08` 的核心)

不是简单"加特征看涨没涨",而是一个能**直接量化 hidden state 边际价值**的消融。两个 head,除输入外完全相同:

- **geom**:动力学 + waypoints + **dist-to-consensus**(给了它共识特征 → 它*能*追平 MB,复现 B);
- **geom+scene**:在 geom 之上 **⊕ 池化的 VLM hidden state**(新信息)。

**scene 的边际 = (geom+scene) − (geom)**,带 paired bootstrap CI,就是"内部状态比纯几何多带多少选择信号"的干净估计。`08` 报告里直接给两个判决:

- `verdict_geomscene_vs_mbr`:verifier 整体打不打得过 MBR;
- `verdict_scene_marginal`:**scene 是否真的 ADD 了几何之外的信号**(这条才是科学结论)。

**工具已验证(合成数据,双向都对):** scene 有信号时 → 边际 CI 显著为正、判 "scene ADDS";scene 是噪声时 → 边际 CI 含 0、判 "scene adds NO signal"(无假阳性)。所以一旦接上真 hidden state,这个脚本会诚实地告诉你内部状态到底有没有用。

## 4. 抽什么 / 怎么 dump(`02 --dump-hidden`,**已实现**)

> **实现**:不改模型,在 `load_model` 里对 VLM text decoder(自动解析 `vlm.model.language_model`,可 `--hidden-module` 覆盖)挂 forward hook,保留**最长**那次前向(= prefill,处理完整 images+prompt+CoT),再对 batch×token 做 mean-pool → 每 clip 一个 H 维 `scene_vec`,存 `<split>_hidden.npz`(key `{sample_id}__scene_vec`)。运行时打印向量维度供 sanity check(应 ≈ hidden-size,**不是** vocab-size;若不对,用 `--hidden-module` 指定正确子模块)。
> 命令:`python scripts/02_run_baseline_inference.py --run-name val_cand5_n1000 --split val --num-traj-samples 5 --execute --dump-hidden`(记得 `export HF_HUB_DISABLE_XET=1`,别设 `HF_HOME`)。

候选共享同一次 VLM rollout(CoT 几乎相同),差异只在采出的轨迹。所以第一版特征:

- **场景向量(候选间共享)**:VLM 最后一层在轨迹起始 token / image token 上做 mean-pool → 一个 H 维向量。编码"这是不是 stop/yield 场景、前方有无障碍"。
- **候选几何(逐候选)**:复用 `07/08` 的 waypoints + 动力学 + dist-to-consensus。

dump 伪代码(在 `02_run_baseline_inference.py` 已有循环里加一存,不重写推理):

```python
# --dump-hidden 时,在拿到 extra(= outputs[2],见 nav_utils.py)的同一处:
hidden = model.vlm.last_hidden_state            # [B, T_tok, H]  (确切属性名以模型为准)
scene_vec = hidden[0, traj_start_slice].mean(0).float().cpu().numpy()   # [H]
arrays[f"{sample_id}__scene_vec"] = scene_vec   # 每 clip 一份,候选共享
np.savez(out_dir / f"{split}_hidden.npz", **arrays)
```

> 待确认:HF 系可用 `output_hidden_states=True`;或对喂给 FlowMatching expert 的那层挂 forward hook。**hidden 很大,务必 pool 成一个向量再存**(1000 clip × H 维完全可控)。

## 5. head(`08_train_verifier.py`,已实现)

- 输入:见上;场景向量来自 `--scene <split>_hidden.npz`(key `{sample_id}__scene_vec`)。无该文件时只跑 geom head 作 sanity baseline 并明确标注。
- 目标:clip 内 z-scored ADE(**回归**,选 argmin)——B 已证明回归比"分类最优候选"稳(后者追离群点反噬)。
- 模型:小 MLP(1 隐层,numpy,Adam,**反向传播经数值梯度检验**),k-fold clip-level CV;高维 hidden 用标准化 + L2 + k-fold 控过拟合。
- 评测:first / MBR / geom / geom+scene / oracle,paired bootstrap(geom_vs_mbr、**geomscene_vs_mbr**、**geomscene_vs_geom**),overall + stop/yield,自动 verdict。

运行(dump 完之后):

```bash
python scripts/08_train_verifier.py --run-name val_cand5_n1000 \
    --scene outputs/runs/val_cand5_n1000/baseline/val_hidden.npz
```

## 6. 里程碑

1. **dump**:给 `02` 加 `--dump-hidden`,重跑一次推理(GPU,≈ baseline 成本),产出 `val_hidden.npz`。← 唯一需要 GPU 的一步
2. **train+ablate**:`08` 训 geom 与 geom+scene,出 `verdict_scene_marginal`。← CPU 秒级,脚本已就绪
3. **若 scene 显著**:加 stop/yield 分场景报告 + case study(scene 帮对了哪些 MBR 选错的 clip);若不显著:也是干净结论(选择信号不在可线性读出的 hidden 子空间)。
4. **(可选)v2.1**:把 scene 也喂给 set-pooling(`07` 的结构)做联合;或用 verifier 分数去 guide FlowMatching 采样。

## 7. 诚实预期

这是**唯一**可能真 >MBR 的方向,但不保证:

- **乐观**:场景向量带来纯几何没有的信号,`verdict_scene_marginal` 显著为正、verifier 在 stop/yield 超过 MBR → 干净正结果,故事闭环("几何到顶 → 接内部状态破顶")。
- **保守**:1000 clip 对高维 hidden 易过拟合,或 first-sample 的 CoT 已把场景信息"用掉" → 边际 ≈ 0。那也是有价值的结论,且因为有 geom-only 对照,排除了"特征工程没做好"的质疑。

口径不变:k-fold/LOO + CI,不在 1000 上吹显著 SOTA;仍是**开环 ADE**(面试需主动说明 ≠ 闭环安全)。

## 8. 面试表述

> "我先用线性回归、分类、和置换不变集合网络三种学习方法证明了纯几何选择在 MBR 见顶,所以原则上的下一步是 frozen-VLM hidden-state 上的 learned verifier——就是 best-of-N + reward model 那套 test-time scaling 范式。我把它设计成一个 geom vs geom+scene 的消融,直接量化内部状态比几何多带多少选择信号,head 和评测都写好且数值梯度检验过了,只差在服务器上 dump 一次 hidden state。"

## 9. 首次结果(n=1000)与两手对策

**结果**(`08 --scene val_hidden.npz`,scene_dim=4096):geom head **精确追平 MBR**(26.28% vs 26.35% gap closed);但 geom+scene **显著输 MBR**(−0.054m,CI[−0.099,−0.010]),且 **scene 边际显著为负**(−0.053m,CI[−0.10,−0.01],frac 0.008)→ `verdict_scene_marginal = "scene HURTS"`。

**两层诊断**:
- *表层*:4096 维 shared 特征 + ~800 训练 clip → 维度灾难,head 过拟合训练折、留出折反而更差(对照合成测试:4 维噪声 scene 边际≈0,4096 维则显著为负)。
- *深层*:scene 对一个 clip 的 5 条候选是**同一份** → within-clip argmin 里线性头中它对所有候选同移、**直接抵消**;只能靠 geom×scene 的非线性交互起作用,而交互正是最易过拟合、最难从 1000 clip 学到的——信号被放在了最难用的位置。

**对策一(便宜,CPU,复用现有 dump)**:`08 --scene-pca 24` —— 无监督(label-free,无 target 泄漏)PCA 把 4096 压到 ~24 维,大幅降过拟合。预期把"显著伤害"拉回"打平 MBR";但受 shared 限制,难真正超过。

**对策二(正解,需 GPU 重 dump)**:`02 --dump-expert-hidden` —— 抽扩散 expert 的 `last_hidden_state`(形状 `(候选数, H)`,**每条候选不同**),是模型对每条候选的内部表示,线性可直接利用、绕开 shared 限制。存 `<split>_expert.npz`,`08 --scene <split>_expert.npz` 自动按"逐候选 2D 特征"处理。**这才是真正可能赢 MBR 的一手。**
> 假设:expert 的 batch 行序 = `trajectory_sample_id`;运行时打印形状供核对(行数应 == 每 clip 候选数;不对就用 `--expert-module` 调整)。

**两手对策的真实结果(n=1000,均已跑):**
- ① `--scene-pca 24`:降维**没救回来**,仍显著伤害(scene 边际 −0.092)→ 共享 scene 死透。
- ② `--dump-expert-hidden`(逐候选 expert hidden,dim=2048,fresh run):**无信号**(scene 边际 −0.042,CI[−0.10, 0.01] 跨 0;`verdict = "scene adds NO signal beyond geometry"`)。geom head 这次还略胜 MBR(31.3% vs 29.6%,CI 含 0)。

**最终结论(钉死)**:hidden-state verifier 这条线**整体为负/无效**。根因是系统层面的:**模型对自己 i.i.d. 的扩散样本不暴露任何 per-sample 质量分**,所以任何基于其几何或内部激活的 reranker 都注定 ≈ 或输免费 MBR。这不是没试够,是模型构造决定的天花板。要真超过,只能换信号源(训模型给样本打分 / 闭环或人工 reward),属另一个更大的项目。
