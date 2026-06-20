# Faithful-Alpamayo-1.5 — Inference-Time Trajectory Selection for a Driving VLA

## TL;DR

Alpamayo 1.5 (a 10B vision-language-action driving model) samples several candidate
trajectories per decision but **deploys the first one**. On 1,000 held-out clips I show
this leaves ~51% *selection* error on the table, and that a simple, training-free
**consensus (Minimum-Bayes-Risk) selector** recovers ~26% of it — significantly — while a
hand-designed *reasoning-aware* selector does **not** help (it even underperforms a
reasoning-blind baseline). The project is a rigorous diagnosis + a small, honest, free
improvement, not a SOTA claim.

## Setup

- **Model**: `nvidia/Alpamayo-1.5-10B`, bf16, SDPA attention, single H100, **no fine-tuning**.
- **Data**: `nvidia/PhysicalAI-Autonomous-Vehicles`, **official val split**. Reproducible
  1,000-clip subset (valid clips with the 4 required cameras + egomotion), drawn by
  proportional **stratified sampling** over driving conditions (daypart × platform). Sensor
  data is **streamed** per clip (no disk/quota footprint).
- **Candidates**: 5 trajectories/clip via the model's own FlowMatching diffusion expert.
  Marginal cost ≈ **1.4×** a single sample (one shared VLM rollout + cheap diffusion draws).

## Diagnosis (the core finding)

I decompose Alpamayo's trajectory error into **generation** error (a good trajectory was
never sampled) vs **selection** error (it was sampled but not chosen), via an oracle
best-of-N analysis (oracle uses GT only for analysis, never for selection):

| | first-sample (deployed) | oracle best-of-5 | recoverable selection error |
| --- | --- | --- | --- |
| ADE (m) | 1.880 | 0.924 | **~51%** |
| FDE (m) | 5.693 | 2.839 | **~50%** |

So roughly half the error is *selection* error — the model already samples a much better
trajectory but doesn't pick it. The first sample is statistically indistinguishable from a
random candidate (1.880 vs 1.865), so the gap is genuine, not luck.

**Key systems insight:** the candidates are i.i.d. diffusion samples and the model exposes
**no per-trajectory score** (the VLM logprob scores the CoT text, which is near-identical
across candidates; the diffusion sampler returns no likelihood). Selection must therefore
come from an *external* signal.

## Method

Four training-free selectors, none of which use GT:

- **reasoning-blind**: pick the smoothest / most central candidate (dynamics only).
- **reasoning-aware**: intent-conditioned scoring from the CoT (stop/yield → prefer
  deceleration & low final speed; turn → correct heading; etc.) + comfort penalty. The
  *aware vs blind* contrast is a **faithfulness probe**: does the stated reasoning carry
  actionable selection information beyond kinematics?
- **consensus / MBR**: pick the candidate nearest the mean of the N samples — a
  variance-reduction / minimum-Bayes-risk estimate (the trajectory analog of LLM
  self-consistency).

## Results (1,000 val clips, 95% bootstrap CIs)

| Selector | ADE (m) | FDE (m) | ADE gap closed | improvement vs first (ADE, 95% CI) |
| --- | --- | --- | --- | --- |
| first-sample (deployed) | 1.880 | 5.693 | 0% (baseline) | — |
| reasoning-aware | 1.815 | 5.605 | +7% | +0.07m, CI [−0.02, 0.16] — **not significant** |
| reasoning-blind | 1.772 | 5.445 | +11% | (crude "central" heuristic) |
| **consensus (MBR)** | **1.635** | **4.964** | **+26%** | **+0.25m, CI [0.18, 0.32]**, 100% > 0 |
| oracle best-of-5 | 0.924 | 2.839 | 100% (upper bound) | — |

- **Consensus is strongly significant** on ADE (CI [0.18, 0.32]) and FDE (+0.73m, CI
  [0.49, 0.97]); bootstrap frac > 0 = **1.00**; win/loss/tie vs first-sample = **498/305/197**.
- **Stop/yield subset (n=212):** consensus is also significant — ADE +0.27m, CI [0.12, 0.42];
  FDE +0.82m, CI [0.37, 1.34]; +27% gap closed.
- **Reasoning is not faithful for selection.** reasoning-aware is **not significant** and
  **underperforms reasoning-blind** (+7% vs +11%; loses head-to-head 60/80; and on the
  stop/yield subset it is *significantly worse* on FDE). Conditioning on the CoT intent adds
  no selection signal — and mildly hurts. The useful signal is plain *centrality/consensus*.

## Mechanism (case studies)

- **Consensus wins** when the deployed first sample is a bad outlier — e.g. a "keep lane"
  clip with first-sample ADE 4.50 m → consensus 0.79 m (oracle 0.37). It avoids the model's
  occasional wild draw.
- **Consensus loses** when the first sample is already excellent — e.g. 0.46 m → 4.21 m:
  averaging pulls away from a lucky-good sample. Net effect is positive and significant; this
  is the honest failure mode.

## What I claim (and what I do not)

- **Claim**: "Alpamayo's deployed first-sample leaves ~51% recoverable selection error; a
  training-free MBR/consensus selector recovers ~26% (ADE) / ~26% (FDE) of it, significantly,
  at ~1.4× inference cost — because the model's diffusion samples carry no internal quality
  score, so the useful signal is sample *centrality*, not the stated reasoning."
- **Negative results (kept honest):** a reasoning-aware heuristic looked promising on a tiny
  9-clip stop/yield subset (+24%) but **vanished/flipped once scaled** (−13% at n=61, still
  negative at n=212) — a cautionary tale I caught only by scaling to 1,000 clips. Reasoning
  does not help selection; it loses to a reasoning-blind baseline.
- **Not claimed:** improving the generator, fine-tuning the 10B, a world model, or RLHF.

## Limitations & next steps

- The improvement is **selection-bounded by the oracle** (recovers ~26% of a ~51% gap);
  modest by design. Cheap post-hoc dynamics/intent features carry little selection signal —
  capturing more likely needs the model's visual context or internal state.
- **Next (Tier 2):** a tiny **learned scorer** (train split / leave-one-clip-out CV; features
  incl. distance-to-consensus, dynamics) to predict the best candidate, tested against free
  MBR. Honest expectation: cheap features ≈ MBR; beating it needs richer (visual/internal)
  features, which could later guide the FlowMatching sampler.

## Reproduce

```bash
N=1000 RUN=val_cand5_n1000 bash scripts/run_baseline.sh   # val subset + multi-candidate inference (GPU)
python scripts/03c_oracle_gap.py  --run-name val_cand5_n1000   # oracle-gap decomposition
python scripts/04_rerank.py       --run-name val_cand5_n1000   # selectors + bootstrap CIs
python scripts/05_case_studies.py --run-name val_cand5_n1000   # win/loss case studies + plots
```
