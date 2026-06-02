#!/usr/bin/env bash
# Source this file on the server before running Alpamayo scripts:
#   source scripts/server_env.sh

set -euo pipefail

export FAITHFUL_ALPAMAYO_PROJECT_ROOT="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/Faithful-Alpamayo-1.5"
export FAITHFUL_ALPAMAYO_DATA_ROOT="${FAITHFUL_ALPAMAYO_PROJECT_ROOT}/data"

# Model and processor files downloaded through transformers/huggingface_hub.
# Do not use this directory for project clips, manifests, outputs, or checkpoints.
export HF_HUB_CACHE="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/huggingface_cache/hub"

# Keep dataset and downstream-library caches inside the gitignored project data directory.
export HF_DATASETS_CACHE="${FAITHFUL_ALPAMAYO_DATA_ROOT}/hf_datasets_cache"
export HF_ASSETS_CACHE="${FAITHFUL_ALPAMAYO_DATA_ROOT}/hf_assets_cache"
export HF_XET_CACHE="${FAITHFUL_ALPAMAYO_DATA_ROOT}/hf_xet_cache"
export PHYSICAL_AI_AV_CACHE_DIR="${FAITHFUL_ALPAMAYO_DATA_ROOT}/physical_ai_av_cache"
export PHYSICAL_AI_AV_LOCAL_DIR="${FAITHFUL_ALPAMAYO_DATA_ROOT}/physical_ai_av_local"

mkdir -p \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_ASSETS_CACHE}" \
  "${HF_XET_CACHE}" \
  "${PHYSICAL_AI_AV_CACHE_DIR}" \
  "${PHYSICAL_AI_AV_LOCAL_DIR}" \
  "${FAITHFUL_ALPAMAYO_DATA_ROOT}/clips" \
  "${FAITHFUL_ALPAMAYO_DATA_ROOT}/manifests" \
  "${FAITHFUL_ALPAMAYO_DATA_ROOT}/outputs/baseline" \
  "${FAITHFUL_ALPAMAYO_DATA_ROOT}/outputs/lora" \
  "${FAITHFUL_ALPAMAYO_DATA_ROOT}/outputs/metrics" \
  "${FAITHFUL_ALPAMAYO_DATA_ROOT}/checkpoints" \
  "${FAITHFUL_ALPAMAYO_DATA_ROOT}/adapters/lora"

echo "FAITHFUL_ALPAMAYO_PROJECT_ROOT=${FAITHFUL_ALPAMAYO_PROJECT_ROOT}"
echo "FAITHFUL_ALPAMAYO_DATA_ROOT=${FAITHFUL_ALPAMAYO_DATA_ROOT}"
echo "HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "PHYSICAL_AI_AV_CACHE_DIR=${PHYSICAL_AI_AV_CACHE_DIR}"
