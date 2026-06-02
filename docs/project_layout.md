# Project Layout

All generated artifacts live inside the project directory and are ignored by
git. Code, configs, and docs stay tracked.

## Tracked

```text
configs/
docs/
faithful_vla/
notebooks/
scripts/
src/
```

## Ignored Runtime Directories

```text
models/
  huggingface/
    hub/        # Alpamayo, Qwen processor, and other HF model repos
    xet/        # HF Xet chunk cache for model downloads

data/
  clips/        # selected/extracted project clips
  manifests/    # split manifests and storage reports
  physical_ai_av/
    cache/      # physical_ai_av HF cache_dir
    local/      # physical_ai_av local_dir, if downloads are materialized

outputs/
  baseline/
  lora/
  metrics/
  figures/
  qualitative_cases/
  hf_assets_cache/

checkpoints/
  lora/         # training checkpoints, not full base-model copies

adapters/
  lora/         # final LoRA adapters

logs/
```

## Server Environment

On the server, run:

```bash
cd /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/Faithful-Alpamayo-1.5
source scripts/server_env.sh
```

`scripts/server_env.sh` sets model downloads to `./models` and dataset/cache
paths to `./data`. It does not put checkpoints or adapters under `data`.
