#!/usr/bin/env python
"""Run Alpamayo 1.5 baseline inference over manifest records.

The script is safe by default: without ``--execute`` it only prints the selected
records and output paths. Model and dataset packages are imported only inside
the execute path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faithful_vla.baseline import (
    build_dry_run_summary,
    build_prediction_row,
    load_manifest_records,
    make_sample_id,
    select_manifest_records,
    text_extra_value,
)
from faithful_vla.run_paths import baseline_output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/manifest_300clips.jsonl"),
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Write default outputs under outputs/runs/<run-name>/baseline/.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually load model/data and run inference.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing run's predictions/trajectories (default: refuse, to protect prior runs).",
    )

    parser.add_argument("--model-id", default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")

    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--num-traj-sets", type=int, default=1)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--num-frames-per-camera", type=int, default=4)

    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--trajectories-npz", type=Path, default=None)
    parser.add_argument("--runtime-jsonl", type=Path, default=None)

    # 2.0 verifier: dump a pooled VLM hidden-state "scene" vector per clip.
    parser.add_argument(
        "--dump-hidden",
        action="store_true",
        help="Also save a pooled VLM hidden-state 'scene' vector per clip for the verifier head (08).",
    )
    parser.add_argument(
        "--hidden-module",
        default=None,
        help="Dotted submodule to hook for hidden states (default: auto-detect the VLM text decoder).",
    )
    parser.add_argument("--hidden-npz", type=Path, default=None)

    # v2.1: per-candidate diffusion-expert hidden state (directly discriminative, unlike the
    # candidate-shared scene vector).
    parser.add_argument(
        "--dump-expert-hidden",
        action="store_true",
        help="Also save the diffusion expert's per-candidate hidden state (N x H per clip).",
    )
    parser.add_argument("--expert-module", default="expert",
                        help="Dotted submodule for the diffusion expert (default: 'expert').")
    parser.add_argument("--expert-npz", type=Path, default=None)
    return parser.parse_args()


def write_jsonl_row(stream: Any, row: dict[str, Any]) -> None:
    stream.write(json.dumps(row, sort_keys=True) + "\n")
    stream.flush()


def torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    if dtype_name == "float16":
        return torch_module.float16
    return torch_module.float32


class HiddenCapture:
    """Forward hook that keeps the pooled last-hidden-state of the VLM's prefill pass.

    During ``vlm.generate`` the hooked text-decoder runs once over the full prompt
    (images + text -- the longest forward) then once per decoded token. We keep the
    longest pass (the prefill = the scene/context representation) and mean-pool it over
    batch and tokens into a single hidden-size vector. This is the candidate-shared
    "scene" feature consumed by the verifier head (scripts/08_train_verifier.py).
    """

    def __init__(self) -> None:
        self._best = None
        self._best_tokens = -1

    def __call__(self, module: Any, inputs: Any, output: Any) -> None:
        h = getattr(output, "last_hidden_state", None)
        if h is None:
            h = output[0] if isinstance(output, (tuple, list)) else output
        try:
            if hasattr(h, "dim") and h.dim() == 3 and int(h.shape[1]) > self._best_tokens:
                self._best_tokens = int(h.shape[1])
                self._best = h.detach().float().mean(dim=(0, 1)).cpu().numpy()
        except Exception:
            pass  # never let hidden capture break inference

    def take(self):
        v, self._best, self._best_tokens = self._best, None, -1
        return v


class ExpertCapture:
    """Hook on the diffusion expert that keeps the LAST forward's per-candidate hidden state.

    The expert runs once per diffusion denoising step; the last call is ~ the final (cleanest)
    step. Its last_hidden_state is (b*, Tf, H) with b* = the candidates; we mean-pool over the Tf
    trajectory tokens -> (b*, H) = one vector per candidate, ordered by trajectory_sample_id. This
    feature differs per candidate (unlike the shared scene vector), so a verifier head can use it
    directly. (Assumes the expert batch order matches trajectory_sample_id; the runtime shape
    print lets you confirm rows == #candidates.)
    """

    def __init__(self) -> None:
        self._last = None

    def __call__(self, module: Any, inputs: Any, output: Any) -> None:
        h = getattr(output, "last_hidden_state", None)
        if h is None:
            h = output[0] if isinstance(output, (tuple, list)) else output
        try:
            if hasattr(h, "dim") and h.dim() == 3:
                self._last = h.detach().float().mean(dim=1).cpu().numpy()  # (b*, H)
        except Exception:
            pass

    def take(self):
        v, self._last = self._last, None
        return v


def resolve_vlm_module(model: Any, dotted: str | None):
    """Return (submodule, path) of the VLM text decoder to hook.

    ``self.vlm`` is a Qwen3VLForConditionalGeneration; its text decoder is the module
    that emits per-token hidden states. We try a configurable path first, then sensible
    fallbacks. The user can override with --hidden-module if the printed shape looks wrong
    (the pooled vector should be hidden-size ~= a few thousand, NOT vocab-size).
    """
    candidates = [dotted] if dotted else [
        "vlm.model.language_model", "vlm.language_model", "vlm.model", "vlm",
    ]
    for path in candidates:
        if not path:
            continue
        obj, ok = model, True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj, path
    raise AttributeError(f"Could not resolve a VLM submodule to hook; tried {candidates}")


def build_avdi() -> Any:
    import physical_ai_av

    avdi_kwargs: dict[str, str] = {}
    cache_dir = os.environ.get("PHYSICAL_AI_AV_CACHE_DIR")
    local_dir = os.environ.get("PHYSICAL_AI_AV_LOCAL_DIR")
    if cache_dir:
        avdi_kwargs["cache_dir"] = str(Path(cache_dir).expanduser())
    if local_dir:
        avdi_kwargs["local_dir"] = str(Path(local_dir).expanduser())
    return physical_ai_av.PhysicalAIAVDatasetInterface(**avdi_kwargs)


def resolve_camera_features(avdi: Any, record: dict[str, Any]) -> list[Any] | None:
    names = record.get("required_cameras") or []
    if not names:
        return None
    return [getattr(avdi.features.CAMERA, name) for name in names]


def load_model(args: argparse.Namespace, torch: Any) -> tuple[Any, Any, Any, Any]:
    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    dtype = torch_dtype(torch, args.dtype)
    model = Alpamayo1_5.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    capture = None
    if getattr(args, "dump_hidden", False):
        capture = HiddenCapture()
        module, path = resolve_vlm_module(model, args.hidden_module)
        module.register_forward_hook(capture)
        print(f"[dump-hidden] hooked VLM submodule: {path}")
    expert_cap = None
    if getattr(args, "dump_expert_hidden", False):
        expert_cap = ExpertCapture()
        module, path = resolve_vlm_module(model, args.expert_module)
        module.register_forward_hook(expert_cap)
        print(f"[dump-expert-hidden] hooked expert submodule: {path}")
    return model, processor, capture, expert_cap


def run_one_record(
    record: dict[str, Any],
    sample_index: int,
    args: argparse.Namespace,
    avdi: Any,
    model: Any,
    processor: Any,
    torch: Any,
    capture: Any = None,
    expert_cap: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], Any, Any]:
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started_at = time.perf_counter()
    data = load_physical_aiavdataset(
        record["clip_id"],
        t0_us=record["t0_us"],
        avdi=avdi,
        maybe_stream=True,
        camera_features=resolve_camera_features(avdi, record),
        num_frames=args.num_frames_per_camera,
    )
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
        num_frames_per_camera=args.num_frames_per_camera,
        nav_text=record.get("nav_text"),
    )
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
    model_inputs = helper.to_device(model_inputs, args.device)

    dtype = torch_dtype(torch, args.dtype)
    device_type = "cuda" if args.device.startswith("cuda") else args.device
    autocast_context = (
        torch.autocast(device_type, dtype=dtype) if args.device.startswith("cuda") else nullcontext()
    )
    with torch.no_grad(), autocast_context:
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            num_traj_sets=args.num_traj_sets,
            max_generation_length=args.max_generation_length,
            return_extra=True,
        )

    runtime_sec = time.perf_counter() - started_at
    max_cuda_memory_gb = None
    if args.device.startswith("cuda"):
        max_cuda_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    arrays: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    sample_id = make_sample_id(record, sample_index)
    arrays[f"{sample_id}__gt_xyz"] = data["ego_future_xyz"].cpu().float().numpy()[0, 0]
    arrays[f"{sample_id}__gt_rot"] = data["ego_future_rot"].cpu().float().numpy()[0, 0]

    pred_xyz_np = pred_xyz.detach().cpu().float().numpy()
    pred_rot_np = pred_rot.detach().cpu().float().numpy()
    for set_index in range(args.num_traj_sets):
        for traj_index in range(args.num_traj_samples):
            trajectory_sample_id = set_index * args.num_traj_samples + traj_index
            row = build_prediction_row(
                record=record,
                sample_index=sample_index,
                trajectory_sample_id=trajectory_sample_id,
                cot=text_extra_value(extra, "cot", set_index, traj_index),
                meta_action=text_extra_value(extra, "meta_action", set_index, traj_index),
                answer=text_extra_value(extra, "answer", set_index, traj_index),
                runtime_sec=runtime_sec,
                max_cuda_memory_gb=max_cuda_memory_gb,
            )
            trajectory_key = row["trajectory_npz_key"]
            row["trajectory_set_index"] = set_index
            row["trajectory_index"] = traj_index
            row["pred_xyz_npz_key"] = f"{trajectory_key}__pred_xyz"
            row["pred_rot_npz_key"] = f"{trajectory_key}__pred_rot"
            row["gt_xyz_npz_key"] = f"{sample_id}__gt_xyz"
            row["gt_rot_npz_key"] = f"{sample_id}__gt_rot"
            arrays[row["pred_xyz_npz_key"]] = pred_xyz_np[0, set_index, traj_index]
            arrays[row["pred_rot_npz_key"]] = pred_rot_np[0, set_index, traj_index]
            rows.append(row)

    runtime_row = {
        "clip_id": record["clip_id"],
        "sample_id": sample_id,
        "split": record["split"],
        "t0_us": record["t0_us"],
        "status": "ok",
        "runtime_sec": runtime_sec,
        "max_cuda_memory_gb": max_cuda_memory_gb,
    }
    scene_vec = capture.take() if capture is not None else None
    expert_mat = expert_cap.take() if expert_cap is not None else None
    return rows, arrays, runtime_row, scene_vec, expert_mat


def run_execute(
    selected_records: list[dict[str, Any]],
    args: argparse.Namespace,
    output_jsonl: Path,
    trajectories_npz: Path,
    runtime_jsonl: Path,
    hidden_npz: Path | None = None,
    expert_npz: Path | None = None,
) -> None:
    import numpy as np
    import torch

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    trajectories_npz.parent.mkdir(parents=True, exist_ok=True)
    runtime_jsonl.parent.mkdir(parents=True, exist_ok=True)

    avdi = build_avdi()
    model, processor, capture, expert_cap = load_model(args, torch)
    all_arrays: dict[str, Any] = {}
    hidden_arrays: dict[str, Any] = {}
    expert_arrays: dict[str, Any] = {}

    with output_jsonl.open("w", encoding="utf-8") as predictions_stream, runtime_jsonl.open(
        "w", encoding="utf-8"
    ) as runtime_stream:
        for sample_index, record in enumerate(selected_records):
            try:
                rows, arrays, runtime_row, scene_vec, expert_mat = run_one_record(
                    record=record,
                    sample_index=sample_index,
                    args=args,
                    avdi=avdi,
                    model=model,
                    processor=processor,
                    torch=torch,
                    capture=capture,
                    expert_cap=expert_cap,
                )
            except Exception as exc:
                runtime_row = {
                    "clip_id": record["clip_id"],
                    "sample_id": make_sample_id(record, sample_index),
                    "split": record["split"],
                    "t0_us": record["t0_us"],
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                write_jsonl_row(runtime_stream, runtime_row)
                if not args.continue_on_error:
                    raise
                continue

            for row in rows:
                write_jsonl_row(predictions_stream, row)
            write_jsonl_row(runtime_stream, runtime_row)
            all_arrays.update(arrays)
            if scene_vec is not None:
                if not hidden_arrays:
                    print(f"[dump-hidden] scene vector dim = {tuple(scene_vec.shape)} "
                          f"(should be hidden-size ~ a few thousand, NOT vocab-size)")
                hidden_arrays[f"{make_sample_id(record, sample_index)}__scene_vec"] = scene_vec
            if expert_mat is not None:
                if not expert_arrays:
                    print(f"[dump-expert-hidden] expert hidden shape = {tuple(expert_mat.shape)} "
                          f"(rows should == #candidates per clip)")
                expert_arrays[f"{make_sample_id(record, sample_index)}__expert_hidden"] = expert_mat
            print(
                f"[{sample_index + 1}/{len(selected_records)}] "
                f"{record['clip_id']} runtime={runtime_row['runtime_sec']:.2f}s"
            )

    np.savez_compressed(trajectories_npz, **all_arrays)
    if getattr(args, "dump_hidden", False) and hidden_arrays and hidden_npz is not None:
        hidden_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(hidden_npz, **hidden_arrays)
        print(f"Wrote hidden scene vectors: {hidden_npz}  ({len(hidden_arrays)} clips)")
    if getattr(args, "dump_expert_hidden", False) and expert_arrays and expert_npz is not None:
        expert_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(expert_npz, **expert_arrays)
        print(f"Wrote expert hidden states: {expert_npz}  ({len(expert_arrays)} clips)")


def main() -> int:
    args = parse_args()
    default_paths = baseline_output_paths(split=args.split, run_name=args.run_name)
    output_jsonl = args.output_jsonl or default_paths["predictions"]
    trajectories_npz = args.trajectories_npz or default_paths["trajectories"]
    runtime_jsonl = args.runtime_jsonl or default_paths["runtime"]
    hidden_npz = args.hidden_npz or (trajectories_npz.parent / f"{args.split}_hidden.npz")
    expert_npz = args.expert_npz or (trajectories_npz.parent / f"{args.split}_expert.npz")

    records = load_manifest_records(args.manifest)
    selected_records = select_manifest_records(records, split=args.split, limit=args.limit)
    if not selected_records:
        raise SystemExit(f"No records selected for split={args.split!r}")

    summary = build_dry_run_summary(selected_records, split=args.split, execute=args.execute)
    summary["manifest"] = str(args.manifest)
    summary["run_name"] = args.run_name
    summary["output_jsonl"] = str(output_jsonl)
    summary["trajectories_npz"] = str(trajectories_npz)
    summary["runtime_jsonl"] = str(runtime_jsonl)
    if args.dump_hidden:
        summary["hidden_npz"] = str(hidden_npz)
    if args.dump_expert_hidden:
        summary["expert_npz"] = str(expert_npz)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not args.execute:
        print("Dry run only. Re-run with --execute to load Alpamayo and PhysicalAI-AV.")
        return 0

    if output_jsonl.exists() and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing predictions: {output_jsonl}\n"
            f"  -> use a NEW --run-name (recommended; leaves your baseline run untouched),\n"
            f"     or pass --overwrite to replace it on purpose.")

    run_execute(selected_records, args, output_jsonl, trajectories_npz, runtime_jsonl,
                hidden_npz=hidden_npz if args.dump_hidden else None,
                expert_npz=expert_npz if args.dump_expert_hidden else None)
    print(f"Wrote predictions: {output_jsonl}")
    print(f"Wrote trajectories: {trajectories_npz}")
    print(f"Wrote runtime: {runtime_jsonl}")
    if args.dump_hidden:
        print(f"Wrote hidden scene vectors: {hidden_npz}")
    if args.dump_expert_hidden:
        print(f"Wrote expert hidden states: {expert_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
