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

## Runtime Directories

```text
data/
  clips/        # selected/extracted project clips
  manifests/    # split manifests and storage reports
  physical_ai_av/
    cache/      # physical_ai_av HF cache_dir
    local/      # physical_ai_av local_dir, if downloads are materialized
  hf_datasets_cache/
  hf_assets_cache/
```

`scripts/server_env.sh` only creates `data/` subdirectories needed for
PhysicalAI-AV access. Other ignored top-level directories such as `outputs/`,
`checkpoints/`, `adapters/`, `models/`, and `logs/` are not created by default.
They should only be created by the specific script that actually writes there.

## Server Environment

On the server, run:

```bash
cd /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/Faithful-Alpamayo-1.5
source scripts/server_env.sh
```

`scripts/server_env.sh` points Hugging Face model/cache reads to the existing
shared cache at `/dss/dssfs05/pn39qo/pn39qo-dss-0001/huggingface/hub`.
Project-generated dataset files stay under `./data`.

For fragile interactive sessions, pre-download model files before running
inference:

```bash
bash scripts/download_model_slow.sh
```

This uses the same model cache path but lowers Hugging Face Xet concurrency.
