# Phase 0 Repository Inspection

This document records the local, code-only inspection of the imported NVIDIA
Alpamayo 1.5 release. No model weights, datasets, example data, training jobs, or
inference scripts were run locally.

## Local Workflow Constraint

- Local machine: code edits, documentation, git commits, and pushes only.
- Server: environment setup, Hugging Face authentication, data access, model
  download, inference, metrics, and training.
- Any script that could download data or model weights must require an explicit
  server-side action.

## Imported Source

- Official source imported from `NVlabs/alpamayo1.5`.
- Upstream clone observed at short commit `a5bd40c`.
- Local repository was initialized without upstream git history.
- Initial local import commit: `3a0e0e2 chore: import alpamayo 1.5 source`.

## Files Inspected

- `README.md`
- `pyproject.toml`
- `notebooks/inference.ipynb`
- `notebooks/inference_nav.ipynb`
- `notebooks/inference_cam_num.ipynb`
- `notebooks/inference_vqa.ipynb`
- `src/alpamayo1_5/test_inference.py`
- `src/alpamayo1_5/load_physical_aiavdataset.py`
- `src/alpamayo1_5/helper.py`
- `src/alpamayo1_5/nav_utils.py`
- `src/alpamayo1_5/config.py`
- `src/alpamayo1_5/models/base_model.py`
- `src/alpamayo1_5/models/alpamayo1_5.py`
- `src/alpamayo1_5/models/token_utils.py`

## Environment Expectations

The official project targets Python 3.12 and pins the core stack in
`pyproject.toml`:

- `torch==2.8.0`
- `transformers==4.57.1`
- `physical-ai-av==0.2.0`
- `flash-attn>=2.8.3`
- `accelerate`, `pandas`, `av`, `einops`, `hydra-core`, `pillow`,
  `matplotlib`, and `seaborn`

The README states that the model and dataset are gated Hugging Face resources:

- Model: `nvidia/Alpamayo-1.5-10B`
- Dataset: `nvidia/PhysicalAI-Autonomous-Vehicles`

The official `src/alpamayo1_5/test_inference.py` should not be run locally for
this project workflow. The README says it downloads example data and the model
weights, with the model weights around 22 GB.

## Model Loading Path

The official examples load the model with:

```python
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

model = Alpamayo1_5.from_pretrained(
    "nvidia/Alpamayo-1.5-10B",
    dtype=torch.bfloat16,
).to("cuda")
```

If `flash-attn` is unavailable, the README documents an SDPA fallback:

```python
model = Alpamayo1_5.from_pretrained(
    "nvidia/Alpamayo-1.5-10B",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda")
```

This call can trigger Hugging Face model/config downloads and must only be used
on the server.

## Dataset Loading Path

The official loader is:

```python
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
```

The loader creates or accepts a `physical_ai_av.PhysicalAIAVDatasetInterface`
and calls `get_clip_feature(..., maybe_stream=True)` for egomotion and camera
features. This can stream from Hugging Face and must only be used on the server.
In this fork, the loader also honors `PHYSICAL_AI_AV_CACHE_DIR` and
`PHYSICAL_AI_AV_LOCAL_DIR` when it creates the dataset interface, so server-side
dataset cache and downloaded clip chunks can live under the gitignored project
`data/` directory. Model downloads are configured separately under `models/`;
checkpoints, adapters, logs, and outputs each have their own top-level ignored
directories.

Default camera features:

- `CAMERA_CROSS_LEFT_120FOV`
- `CAMERA_FRONT_WIDE_120FOV`
- `CAMERA_CROSS_RIGHT_120FOV`
- `CAMERA_FRONT_TELE_30FOV`

Default camera index mapping:

- `0`: cross-left 120 FOV
- `1`: front-wide 120 FOV
- `2`: cross-right 120 FOV
- `6`: front-tele 30 FOV

The camera-count notebook confirms flexible camera counts with one, two, and
four camera configurations.

## Expected Input Format

`load_physical_aiavdataset` returns a dictionary with:

- `image_frames`: tensor shaped `(N_cameras, num_frames, 3, H, W)`.
- `camera_indices`: tensor shaped `(N_cameras,)`.
- `ego_history_xyz`: tensor shaped `(1, 1, 16, 3)` by default.
- `ego_history_rot`: tensor shaped `(1, 1, 16, 3, 3)` by default.
- `ego_future_xyz`: tensor shaped `(1, 1, 64, 3)` by default.
- `ego_future_rot`: tensor shaped `(1, 1, 64, 3, 3)` by default.
- `relative_timestamps`: tensor shaped `(N_cameras, num_frames)`.
- `absolute_timestamps`: tensor shaped `(N_cameras, num_frames)`.
- `t0_us`
- `clip_id`

The default temporal setup is 16 history steps, 64 future steps, and a 0.1 s
time step. This matches a 6.4 s future trajectory horizon at 10 Hz.

`helper.create_message` constructs the chat prompt. It flattens camera frames,
adds camera names and frame numbers when `camera_indices` is provided, inserts
history trajectory placeholder tokens, optionally inserts a navigation span, and
starts the assistant response with `<|cot_start|>`.

The standard inference input is built as:

```python
messages = helper.create_message(
    frames=data["image_frames"].flatten(0, 1),
    camera_indices=data["camera_indices"],
)
processor = helper.get_processor(model.tokenizer)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)
model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}
```

Navigation conditioning is passed through `helper.create_message(...,
nav_text="Turn right in 30m")`. The navigation utilities also provide a
classifier-free guidance path through
`sample_trajectories_from_data_with_vlm_rollout_cfg_nav`.

## Output Format

Primary inference API:

```python
pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
    data=model_inputs,
    top_p=0.98,
    temperature=0.6,
    num_traj_samples=1,
    max_generation_length=256,
    return_extra=True,
)
```

`pred_xyz` is rearranged to `(B, num_traj_sets, num_traj_samples, T, 3)`.
`pred_rot` is rearranged to `(B, num_traj_sets, num_traj_samples, T, 3, 3)`.
With `return_extra=True`, `extra` contains text fields extracted from generated
tokens and reshaped to `(B, num_traj_sets, num_traj_samples)`.

Known `extra` keys from `token_utils.extract_text_tokens`:

- `cot`
- `meta_action`
- `answer`

The official test script prints `extra["cot"][0]`. For this project, downstream
wrappers should normalize this into one JSONL row per decision sample and
trajectory sample.

## Where CoC Reasoning Is Generated

`Alpamayo1_5.sample_trajectories_from_data_with_vlm_rollout` runs:

1. `self.fuse_traj_tokens` to replace history trajectory placeholders with
   encoded history tokens.
2. `self.vlm.generate` for autoregressive VLM generation.
3. `ExpertLogitsProcessor` to mask trajectory-token logits during CoC text
   generation.
4. `StopAfterEOS`, using `<|traj_future_start|>` as the stopping boundary.
5. `extract_text_tokens` to extract `cot`, `meta_action`, and `answer`.

This means the CoC reasoning is text generated by the VLM before the diffusion
expert samples continuous trajectories.

## Where Trajectory Predictions Are Generated

After VLM generation, the method:

1. Reuses VLM `past_key_values` as prompt cache.
2. Builds expert position IDs and attention masks.
3. Defines a diffusion `step_fn` that projects noisy actions through
   `action_in_proj`, runs the expert model, and maps hidden states through
   `action_out_proj`.
4. Calls `self.diffusion.sample`.
5. Converts sampled actions to trajectories with
   `self.action_space.action_to_traj`.

This is the natural wrapper boundary for baseline inference and metrics. We
should avoid invasive modifications to `src/alpamayo1_5/models/alpamayo1_5.py`
until the baseline output format is stable.

## Dataset Files Needed

The current official loader directly requires:

- Egomotion labels: `avdi.features.LABELS.EGOMOTION`
- Default four camera features listed above

Ground-truth future trajectory is derived from egomotion and returned as
`ego_future_xyz` / `ego_future_rot`. Calibration is not explicitly consumed by
the current loader path, though it may be internal to camera decoding or useful
for future visualization overlays.

Optional future modules may need labels for entity coverage, but entity labels
should remain optional and must not block trajectory or reasoning-action metrics.

## Disk and Runtime Notes

- Official README inference estimate: single-sample inference uses about 24 GB
  VRAM.
- Multi-sample inference with 16 trajectory samples uses about 40 GB VRAM.
- Multi-sample inference with classifier-free guidance uses about 60 GB VRAM.
- Model weights are about 22 GB.
- The project rule remains: if a step would require more than 50 GB additional
  disk, stop and explain before doing it.

## Immediate Engineering Direction

The first code layer should be non-invasive wrappers and config-driven scripts:

- Environment checker that does not load model weights.
- Manifest preparation with dry-run and clip-level split validation.
- Baseline inference wrapper that requires an explicit server-side execute flag
  before importing `physical_ai_av` or loading the model.
- Metrics that consume compact JSONL/NPZ outputs and can be tested locally with
  synthetic arrays.
