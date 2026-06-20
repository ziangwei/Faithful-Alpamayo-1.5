# Tier-3 plan: a learned verifier on the frozen VLM's hidden states

日期:2026-06-20
状态:**设计文档(未运行)**。这是 A/B 之后唯一可能真正打破 MBR 天花板的方向,需要一次 GPU dump,不微调 10B。

## 1. 为什么是这个方向

A 证明了 ~51% 是可恢复的选择误差;免费 MBR 显著捞回 ~26%。B(`06`)与 set-aggregator(`07`)进一步证明:**只用输出几何 + 廉价动力学特征,学习打不过 MBR**——ridge 追平 MBR、logreg 反噬、放开手的 set 网络也只是把"共识"重新学了一遍。结论是天花板不在模型容量,在**信息量**:候选轨迹的 (x,y) 几何里没有"为什么这条更好"的场景依据。

模型内部其实"知道"场景——前视相机里有没有行人、红灯、车道线、要不要让行——但这些信息在采样出 (x,y) 后就被丢掉了。**要打破 MBR,就得把模型的内部状态接出来当特征。** 这正是现代 LLM test-time scaling 的标准范式:**best-of-N 采样 + 一个学出来的 verifier / reward model 给候选打分**。本项目把它搬到驾驶 VLA 上。

## 2. 核心思想(一句话)

> 冻结 Alpamayo;对每条候选轨迹,从模型内部状态取一个特征向量;训一个小 verifier head 预测该候选的质量(ADE 排序),推理时选 argmin。对照免费 MBR,看场景信息能否真的超过纯共识。

不动 10B 主体,只训一个 <100K 参数的 head。与 A/B 同一套评测口径(LOO/k-fold CV + paired bootstrap vs MBR + stop/yield 子集)。

## 3. 抽什么特征(关键,且需在服务器上对模型确认)

候选之间共享同一次 VLM rollout(CoT 文本几乎相同),差异只在扩散 expert 采出的轨迹。所以判别信号要同时包含:

- **场景上下文(候选间共享)**:VLM 最后一层在轨迹起始 token / 池化后的 image token 上的 hidden state。它编码了"这是个 stop/yield 场景吗、前方有没有障碍"。这是几何特征完全没有的东西。
- **候选轨迹嵌入(逐候选不同)**:把每条候选轨迹编码成向量(可复用 `07` 的 waypoints,或扩散 expert 对该样本的中间表示)。

第一版最简单可跑的特征:`per_candidate_vec = [pooled VLM last-hidden-state(场景,共享)] ⊕ [07 的 waypoints+动力学(几何,逐候选)]`。哪怕只是"场景向量 + 已有几何",也已让 head 能学到**场景条件化的选择**(例:靠近斑马线 → 偏好减速那条),这是 aware/blind 启发式当年想做、但用文本 intent 做不到的事。

> 待在服务器上确认的点:`extra` 来自 `model(...)` 的第 3 个返回(见 `nav_utils.py` 的 `outputs[2]`);hidden state 的确切张量要在 `helper.py` 的 generate 路径 / 模型 forward 里定位。HF 系模型可用 `output_hidden_states=True`,或对喂给 FlowMatching expert 的那层挂 forward hook。

## 4. 怎么 dump(在 `02_run_baseline_inference.py` 加一个 flag)

不重写推理,只在已有循环里多存一个张量。伪代码:

```python
# 02_run_baseline_inference.py，--dump-hidden 时:
# 取场景向量(候选间共享):VLM 最后一层、对 trajectory-start / image tokens 做 mean-pool
hidden = model.vlm.last_hidden_state           # [B, T_tok, H]  (确切属性名以模型为准)
scene_vec = hidden[0, traj_start_slice].mean(0).float().cpu().numpy()   # [H]
arrays[f"{sample_id}__scene_vec"] = scene_vec   # 每个 clip 一份,候选共享
# 逐候选向量:复用 07 的 waypoints(已有 pred_xyz),或扩散 expert 中间态(若可取)
```

输出 `outputs/runs/<run>/baseline/<split>_hidden.npz`(key = `sample_id__scene_vec`),与现有 `predictions.jsonl` / `trajectories.npz` 同序对齐。**hidden state 很大,务必 pool 成一个向量再存**(单 clip 一个 H 维向量,1000 clip 完全可控)。

## 5. verifier head + 评测(`08_train_verifier.py`,骨架待写)

结构与 `06`/`07` 同构,只是输入换成 hidden-state 特征:

- 特征:`scene_vec`(广播到 5 条候选)⊕ 逐候选 waypoints/动力学。
- 目标:clip 内 z-scored ADE(回归,选 argmin)——`06` 证明回归比"分类最优候选"稳(后者追离群点反噬)。
- 模型:小 MLP(1–2 隐层,强 L2 / dropout),**k-fold 或 LOO CV**;1000 clip 上务必强正则防过拟合。
- 评测:vs first / vs **MBR**(关键)/ vs oracle,paired bootstrap CI + 输赢,overall + stop/yield。复用 `07` 的 `paired_bootstrap` / `analyze` / `verdict_from_ci`。

## 6. 诚实预期

这是**唯一**可能真正 >MBR 的方向,但不保证:

- **乐观**:场景向量带来纯几何没有的信号,verifier 在 stop/yield 等场景显著超过 MBR → 干净的正结果,且故事完整("廉价特征到顶 → 接内部状态破顶")。
- **保守**:1000 clip 对高维 hidden 特征容易过拟合,或 first-sample 的 CoT 已把场景信息"用掉"了 → verifier ≈ MBR。那也是有价值的结论(选择信号不在可线性读出的 hidden 子空间里)。

无论哪种,口径都和 A/B 一致:k-fold/LOO + CI,不在 1000 上吹显著 SOTA。

## 7. 成本与边界

- 需**重跑一次推理** dump hidden(≈ baseline 的 GPU 成本),之后训 head 是 CPU 秒级。
- 不微调 10B、不接世界模型、不做闭环 RL。
- 仍是**开环 ADE** 口径,结论同样受"ADE ≠ 闭环安全"约束(面试需主动说明)。

## 8. 面试表述

> "我先证明了廉价后验特征已经到了 MBR 的天花板(回归追平、分类反噬、set 网络也只是重学共识),所以原则上的下一步是一个 frozen-VLM hidden-state 上的 learned verifier——就是 best-of-N + reward model 那套 test-time scaling 范式——因为只有模型内部状态才带着输出几何丢掉的场景理解。这是我会做的 Tier-3,不动 10B。"

不声称已实现/已跑;这是有据可依的下一步设计。
