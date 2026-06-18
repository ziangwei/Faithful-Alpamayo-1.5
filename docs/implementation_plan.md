# Implementation Plan

Status: superseded for the main project direction. See
`docs/technical_route_reassessment_zh.md`. The LoRA/RL-heavy route below is kept
as historical context only, not as the recommended next step.

This plan adapts the original project prompt to the actual local workflow:
local code changes only, server execution only.

## Working Contract

- Local: inspect code, write wrappers, configs, metrics, docs, commits, pushes.
- Server: pull, install dependencies, authenticate with Hugging Face, download
  gated resources, run inference, prepare data, train LoRA adapters, and report
  outputs back.
- Local scripts must not download data or model weights by default.
- Generated data, outputs, model weights, checkpoints, videos, and arrays are
  ignored by git.

## Phase 0 Scope

Phase 0 should produce a stable foundation before any baseline run:

1. Document official model and dataset interfaces.
2. Add safe config files for dataset, inference, metrics, perturbations, and
   LoRA.
3. Add a no-download environment checker.
4. Add a dry-run manifest generator that enforces clip-level splits.
5. Add notes for the exact server commands to run next.

## Phase 1: Dataset Subset Preparation

Build around clip IDs, not individual decision samples, so split leakage is
impossible by construction.

Target split:

- Train: 180 clips
- Validation: 60 clips
- Test: 60 clips

Initial server flow:

```bash
python scripts/00_check_env.py --strict
python scripts/01_prepare_subset.py \
  --clip-id-file data/source_clip_ids.txt \
  --write
```

The first implementation writes compact manifest metadata only. It does not
download media. Later, when server data access is confirmed, this script can be
extended with explicit `--execute-copy` or `--execute-download` flags.

## Phase 2: Baseline Inference

Add `scripts/02_run_baseline_inference.py` after manifest shape is stable.

Wrapper requirements:

- Default to dry-run.
- Require explicit `--execute` for any import path that can load model or data.
- Load `Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B")` only on server.
- Save JSONL rows for text and metadata.
- Save trajectories in compact NPZ.
- Keep one row per `(clip_id, sample_id, trajectory_sample_id)`.

Recommended normalized output fields:

- `clip_id`
- `sample_id`
- `t0_us`
- `split`
- `nav_text`
- `camera_indices`
- `cot`
- `meta_action`
- `trajectory_npz_key`
- `runtime_sec`
- `max_cuda_memory_gb`

## Phase 3: Metrics

Metrics should consume saved outputs instead of importing the Alpamayo model.

First metric modules:

- Trajectory ADE/FDE from `pred_xyz` and `ego_future_xyz`.
- Kinematic proxies from predicted xy coordinates.
- Reasoning intent parser over `cot` and `meta_action`.
- Reasoning-action consistency checks with thresholds from
  `configs/metrics.yaml`.

This can be locally unit-tested with synthetic JSONL/NPZ data.

## Phase 4: Failure Mining

Rank failure cases by:

- Reasoning-action inconsistency.
- High ADE/FDE.
- Perturbation sensitivity.
- Missing optional entity mentions when labels are available.

Output should be compact JSONL plus selected frames only after server execution.

## Phase 5: LoRA

Do not assume module names. First add an inspection utility that lists candidate
linear layers from the loaded server model. Then decide LoRA target modules from
the real list.

Conservative starting point:

- Freeze base model.
- Prefer text/VLM attention projections first.
- Rank 8 or 16.
- BF16.
- Save adapters only.
- Never save full model checkpoints.

## Phase 6: Comparison and Reporting

Compare baseline and LoRA outputs on the held-out test split:

- Reasoning-action consistency rate.
- Inconsistency counts by type.
- ADE/FDE when GT is available.
- Perturbation deviation.
- Intent flip rate.
- Entity coverage only if labels are available.

## Commit Strategy

Use small commits:

1. `docs: add phase0 inspection and plan`
2. `chore: add safe configs and local guards`
3. `feat: add dry-run subset manifest builder`
4. `feat: add baseline inference wrapper`
5. `feat: add consistency metrics`
