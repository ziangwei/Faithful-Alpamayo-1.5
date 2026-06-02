#!/usr/bin/env bash
# Slow, controlled model pre-download for servers where direct inference-triggered
# downloads can destabilize interactive sessions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing hf command. Install with: pip install -U huggingface_hub" >&2
  exit 1
fi

echo "Downloading model cache with low concurrency."
echo "Target cache: ${HF_HUB_CACHE}"

hf download nvidia/Alpamayo-1.5-10B --cache-dir "${HF_HUB_CACHE}"
hf download Qwen/Qwen3-VL-2B-Instruct --cache-dir "${HF_HUB_CACHE}"

echo "Model pre-download complete."
