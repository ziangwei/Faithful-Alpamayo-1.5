#!/usr/bin/env bash
# End-to-end baseline for the reranker val set.
#   CPU: build the val clip list + manifest.
#   GPU: multi-candidate Alpamayo inference (5 trajectories per clip).
#
# Run from the repo root, inside your H100 allocation:
#   LIMIT=3 bash scripts/run_baseline.sh   # quick smoke test on 3 clips first
#   bash scripts/run_baseline.sh           # full run (N clips)
#   N=300 RUN=val_cand5_n300 bash scripts/run_baseline.sh   # the final, larger run
set -euo pipefail

export HF_HUB_DISABLE_XET=1                 # cluster can't reach the Xet endpoint; force plain HTTP

N=${N:-60}                                  # how many official-val clips to use
RUN=${RUN:-val_cand5}                       # outputs land in outputs/runs/$RUN/
CLIPS=data/source_clip_ids_proposed.txt
LIMIT_ARG=""; [ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit ${LIMIT}"

echo "==== [1/3 CPU] build val clip list ($N official-val clips) ===="
python scripts/01b_build_val_subset.py --select "$N" --out "$CLIPS" --write --force

echo "==== [2/3 CPU] build manifest (all $N as val) ===="
python scripts/01_prepare_subset.py \
  --clip-id-file "$CLIPS" \
  --train-count 0 --val-count "$N" --test-count 0 \
  --write
MANIFEST=data/manifests/manifest_300clips.jsonl   # 01_prepare_subset always writes this filename

echo "==== [3/3 GPU] multi-candidate inference (5 trajectories/clip) ${LIMIT_ARG:+[SMOKE ${LIMIT}]} ===="
python scripts/02_run_baseline_inference.py \
  --manifest "$MANIFEST" \
  --split val \
  --run-name "$RUN" \
  --num-traj-samples 5 \
  --device cuda \
  --attn-implementation sdpa \
  --continue-on-error \
  $LIMIT_ARG \
  --execute

echo "==== DONE -> outputs/runs/$RUN/baseline/  (val_predictions.jsonl + val_trajectories.npz) ===="
