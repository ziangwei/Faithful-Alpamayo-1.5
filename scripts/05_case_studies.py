#!/usr/bin/env python
"""Pick top consensus(MBR)-win / consensus-loss clips and plot their trajectories.

CPU-only, read-only inputs. Recomputes per-candidate ADE + the consensus (centroid)
selection from the baseline outputs, ranks clips by how much consensus beats the
deployed first-sample, and writes the top wins + losses as a markdown table plus
xy trajectory plots (GT / first-sample / consensus / oracle) for the writeup.

Run (after the baseline run produced outputs/runs/<run>/baseline/):
  python scripts/05_case_studies.py --run-name val_cand5_n300 --top-k 4
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faithful_vla.metrics import compute_trajectory_metrics, parse_intents
from faithful_vla.run_paths import baseline_output_paths


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="val")
    p.add_argument("--run-name", default=None)
    p.add_argument("--predictions", type=Path, default=None)
    p.add_argument("--trajectories", type=Path, default=None)
    p.add_argument("--time-step", type=float, default=0.1)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--figures-dir", type=Path, default=None)
    return p.parse_args()


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    a = parse_args()
    import numpy as np
    bp = baseline_output_paths(split=a.split, run_name=a.run_name)
    preds = read_jsonl(a.predictions or bp["predictions"])
    traj = np.load(a.trajectories or bp["trajectories"])

    groups: dict[str, Any] = {}
    for p in preds:
        pk, gk = p.get("pred_xyz_npz_key"), p.get("gt_xyz_npz_key")
        if pk not in traj or gk not in traj:
            continue
        xyz = np.asarray(traj[pk], dtype=float)
        g = groups.setdefault(p["sample_id"], {"gt": np.asarray(traj[gk], dtype=float), "cands": [],
                                               "cot": p.get("cot"), "meta_action": p.get("meta_action"),
                                               "answer": p.get("answer"), "clip_id": p.get("clip_id")})
        m = compute_trajectory_metrics(xyz.tolist(), g["gt"].tolist(), time_step=a.time_step)
        g["cands"].append({"tid": p.get("trajectory_sample_id", 0), "ade": m["ade_m"], "fde": m["fde_m"], "xyz": xyz})

    cases = []
    for sid, g in groups.items():
        cands = sorted(g["cands"], key=lambda c: c["tid"])
        first = next((c for c in cands if c["tid"] == 0), cands[0])
        oracle = min(cands, key=lambda c: c["ade"])
        centroid = np.mean([c["xyz"][:, :2] for c in cands], axis=0)
        cen = min(cands, key=lambda c: float(np.mean(np.linalg.norm(c["xyz"][:, :2] - centroid, axis=1))))
        intents = sorted(set(parse_intents({"cot": g["cot"], "meta_action": g["meta_action"], "answer": g["answer"]})))
        cases.append({"clip_id": g["clip_id"] or sid, "cot": g["cot"] or "", "intents": intents,
                      "first_ade": first["ade"], "centroid_ade": cen["ade"], "oracle_ade": oracle["ade"],
                      "improvement": first["ade"] - cen["ade"],
                      "gt": g["gt"], "first_xyz": first["xyz"], "centroid_xyz": cen["xyz"], "oracle_xyz": oracle["xyz"]})

    cases.sort(key=lambda c: c["improvement"], reverse=True)
    wins, losses = cases[:a.top_k], cases[-a.top_k:][::-1]

    fig_dir = a.figures_dir or (Path("outputs/runs") / a.run_name / "figures" if a.run_name else Path("figures"))
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        have_mpl = True
    except Exception:
        have_mpl = False

    def plot(case, tag):
        if not have_mpl:
            return None
        fig, ax = plt.subplots(figsize=(5, 5))
        for label, xyz, style in [("GT", case["gt"], "k-"), ("first-sample", case["first_xyz"], "r--"),
                                   ("consensus", case["centroid_xyz"], "g-"), ("oracle", case["oracle_xyz"], "b:")]:
            ax.plot(xyz[:, 0], xyz[:, 1], style, label=label, linewidth=2)
        ax.set_title(f"{tag} {case['clip_id'][:8]} | first {case['first_ade']:.2f} -> cons {case['centroid_ade']:.2f} m")
        ax.legend(fontsize=8); ax.set_aspect("equal", "datalim"); ax.grid(alpha=0.3)
        name = f"{tag}_{case['clip_id'][:8]}.png"
        fig.tight_layout(); fig.savefig(fig_dir / name, dpi=110); plt.close(fig)
        return name

    lines = ["# Consensus (MBR) selection — case studies", "",
             f"run: {a.run_name} | top {a.top_k} wins + losses (consensus vs first-sample, by ADE)", ""]
    for tag, group in [("WIN", wins), ("LOSS", losses)]:
        lines += [f"## {tag}: consensus {'beats' if tag == 'WIN' else 'underperforms'} first-sample", "",
                  "| clip | intents | first ADE | consensus ADE | oracle ADE | cot |",
                  "| --- | --- | --- | --- | --- | --- |"]
        imgs = []
        for c in group:
            img = plot(c, tag); imgs.append(img)
            cot = (c["cot"] or "").replace("|", "/").replace("\n", " ")[:70]
            lines.append(f"| {c['clip_id'][:8]} | {','.join(c['intents']) or '-'} | {c['first_ade']:.2f} | {c['centroid_ade']:.2f} | {c['oracle_ade']:.2f} | {cot} |")
        lines.append("")
        for img in imgs:
            if img:
                lines.append(f"![{img}](../figures/{img})")
        lines.append("")

    out = a.out or (Path("outputs/runs") / a.run_name / "analysis" / "case_studies.md" if a.run_name else Path("case_studies.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out}  ({'plots in ' + str(fig_dir) if have_mpl else 'NO matplotlib -> table only'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
