# Faithful-Alpamayo-1.5 — Inference-Time Trajectory Selection for a Driving VLA

## TL;DR

Alpamayo 1.5 (a 10B vision-language-action driving model) samples several candidate
trajectories per decision but **deploys the first one**. I show that this leaves a large,
quantifiable amount of *selection* error on the table, and that a simple, training-free
**consensus (Minimum-Bayes-Risk) selector** recovers a statistically significant slice of
it — while a hand-designed *reasoning-aware* selector does **not** help. The project is a
rigorous diagnosis + a small, honest improvement, not a SOTA claim.

## Setup

- **Model**: `nvidia/Alpamayo-1.5-10B`, bf16, SDPA attention, single H100, no fine-tuning.
- **Data**: `nvidia/PhysicalAI-Autonomous-Vehicles`, **official val split**. I built a
  reproducible 300-clip subset (`clip_is_valid`, has the 4 required cameras + egomotion),
  drawn by proportional **stratified sampling** over driving conditions (daypart ×
  platform). Sensor data is **streamed** per clip at inference (no disk/quota footprint).
- **Candidates**: 5 trajectories per clip via the model's own FlowMatching diffusion
  expert (`num_traj_samples=5`). Marginal cost ≈ **1.4×** a single sample (one shared VLM
  rollout + cheap diffusion draws), *not* 5×.

## Diagnosis (the core finding)

I decompose Alpamayo's trajectory error into **generation** error (the good trajectory was
never sampled) vs **selection** error (it was sampled but not chosen), via an oracle
best-of-N analysis (oracle uses GT only for analysis, never for selection):

| | first-sample (deployed) | oracle best-of-5 | recoverable selection error |
| --- | --- | --- | --- |
| ADE (m) | 1.762 | 0.925 | **47%** |
| FDE (m) | 5.524 | 2.949 | **47%** |

So ~47% of the error is *selection* error — the model already generates a much better
trajectory among its samples but doesn't pick it. Crucially, the first sample is
statistically indistinguishable from a random candidate (1.762 vs 1.851), so the gap is
genuine, not luck.

**Key systems insight:** the candidates are i.i.d. diffusion samples and the model exposes
**no per-trajectory score** (the VLM logprob scores the CoT text, which is near-identical
across candidates; the diffusion sampler returns no likelihood). So selection must come
from an *external* signal.

## Method

I evaluate four training-free selectors (none use GT):

- **reasoning-blind**: pick the smoothest / most central candidate (dynamics only).
- **reasoning-aware**: intent-conditioned scoring from the CoT (stop/yield → prefer
  deceleration & low final speed; turn → correct heading; etc.) + comfort penalty. The
  *aware vs blind* contrast is a **faithfulness probe**: does the stated reasoning carry
  actionable information for selection, beyond kinematics?
- **consensus / MBR**: pick the candidate nearest the mean of the N samples — a
  variance-reduction / minimum-Bayes-risk estimate (the trajectory analog of LLM
  self-consistency).

## Results (300 val clips, 95% bootstrap CIs)

| Selector | ADE (m) | FDE (m) | ADE gap closed | improvement vs first (ADE, 95% CI) |
| --- | --- | --- | --- | --- |
| first-sample (deployed) | 1.762 | 5.524 | 0% (baseline) | — |
| reasoning-blind | 1.805 | 5.586 | −5% | not significant |
| reasoning-aware | 1.769 | 5.578 | −1% | **null**: +0.00m, CI [−0.15, +0.15] |
| **consensus (MBR)** | **1.625** | **4.958** | **+16%** | **+0.14m, CI [0.03, 0.25]**, 99% > 0 |
| oracle best-of-5 | 0.925 | 2.949 | 100% (upper bound) | — |

- **Consensus is statistically significant** on both ADE (CI excludes 0) and FDE
  (+0.57m, CI [0.18, 0.96]); win/loss/tie vs first-sample = **148 / 89 / 63**.
- **Reasoning-aware is null** and ≈ identical to reasoning-blind (260/300 ties) → the CoT
  reasoning does **not** add actionable selection signal over kinematics here.
- **Stop/yield subset (n=61):** consensus significant on FDE (CI [0.05, 1.63]), suggestive
  on ADE (mean +0.15m, 89% > 0, but CI crosses 0). Honest: under-powered on ADE at n=61.

## Mechanism (case studies)

- **Consensus wins** when the deployed first sample is a bad outlier — e.g. a "keep lane"
  clip where first-sample ADE 4.50 m → consensus 0.79 m (oracle 0.37). Consensus avoids
  the model's occasional wild draw.
- **Consensus loses** when the first sample is already excellent — e.g. 0.46 m → 4.21 m:
  averaging pulls away from a lucky-good sample. Net effect is positive and significant,
  but this is the honest failure mode.

## What I'd claim (and what I would not)

- **Claim**: "Alpamayo's deployed first-sample leaves ~47% recoverable selection error;
  a training-free MBR/consensus selector recovers ~16% (ADE) / ~22% (FDE) of it,
  significantly, at ~1.4× inference cost — because the model's diffusion samples carry no
  internal quality score."
- **Negative results (kept honest):** a reasoning-aware heuristic looked promising on a
  9-clip stop/yield subset (+24%) but **vanished/flipped at n=61** (−13%) — a cautionary
  tale I caught only by scaling. Reasoning does not help selection beyond consensus.
- **Not claimed:** improving the generator, fine-tuning the 10B, a world model, or RLHF.

## Limitations & next steps

- The improvement is **selection-bounded by the oracle** (recovers ~16% of a 47% gap);
  modest by design. Cheap post-hoc dynamics/intent features don't carry the selection
  signal — likely it needs the model's visual context or internal state.
- **Next (Tier 2):** a tiny **learned scorer** trained on the train split (features incl.
  distance-to-consensus, dynamics, optionally context) to predict the best candidate, and
  test whether it beats free MBR. If it carries real signal, it could later be used as a
  guidance reward inside the FlowMatching sampler.

## Reproduce

```bash
bash scripts/run_baseline.sh                       # build val subset + multi-candidate inference (GPU, ~1h)
python scripts/03c_oracle_gap.py --run-name <run>  # oracle-gap decomposition
python scripts/04_rerank.py      --run-name <run>  # selectors + bootstrap CIs
python scripts/05_case_studies.py --run-name <run> # win/loss case studies + plots
```
