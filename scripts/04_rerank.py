#!/usr/bin/env python
"""Reasoning-aware (and reasoning-blind) inference-time trajectory reranker.

CPU-only, read-only inputs. For each clip's N candidate trajectories it computes
dynamics features, then selects one WITHOUT using GT:
  - aware : intent-conditioned scoring (stop/yield -> prefer decel & low final
            speed; turn -> prefer correct heading; lane-change/avoid -> prefer
            lateral; else -> prefer straight) + a global comfort (jerk) penalty.
  - blind : dynamics only, no intent (prefer smoothest + most central) -- the
            ablation that tells us whether the reasoning actually helps selection.
Optional confidence gating only overrides candidate 0 when the margin is large.

Evaluation (uses GT only to score, never to select): how much of the oracle gap
each reranker closes, vs first-sample, overall and on the stop/yield subset, with
per-case win/loss. Writes a JSON report + a per-clip selection JSONL for case studies.

Run (after the baseline run produced outputs/runs/<run>/baseline/):
  python scripts/04_rerank.py --run-name val_cand5
"""
from __future__ import annotations
import argparse, json, math, statistics, sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faithful_vla.metrics import compute_trajectory_metrics, parse_intents
from faithful_vla.run_paths import baseline_output_paths

STOP_LIKE = {"stop", "yield", "slow_down"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="val")
    p.add_argument("--run-name", default=None)
    p.add_argument("--predictions", type=Path, default=None)
    p.add_argument("--trajectories", type=Path, default=None)
    p.add_argument("--time-step", type=float, default=0.1)
    p.add_argument("--gate-margin", type=float, default=0.0, help="Only override candidate 0 if best score beats it by > margin.")
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples for CIs (0 to skip).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--selection-out", type=Path, default=None)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _mean(xs): return sum(xs) / len(xs) if xs else float("nan")


def candidate_features(xyz: list[list[float]], dt: float) -> dict[str, float]:
    pts = [(float(p[0]), float(p[1])) for p in xyz]
    if len(pts) < 2:
        return dict(final_speed=0.0, speed_delta=0.0, max_abs_jerk=0.0, heading_change=0.0, lateral_disp=0.0)
    speeds = [math.dist(pts[i], pts[i - 1]) / dt for i in range(1, len(pts))]
    final_speed = speeds[-1]
    speed_delta = speeds[0] - speeds[-1]                       # >0 = decelerating
    accels = [(speeds[i] - speeds[i - 1]) / dt for i in range(1, len(speeds))]
    jerks = [(accels[i] - accels[i - 1]) / dt for i in range(1, len(accels))]
    max_abs_jerk = max((abs(j) for j in jerks), default=0.0)
    h0 = math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])
    h1 = math.atan2(pts[-1][1] - pts[-2][1], pts[-1][0] - pts[-2][0])
    heading_change = math.atan2(math.sin(h1 - h0), math.cos(h1 - h0))
    lateral = 0.0
    for p in pts:
        dx, dy = p[0] - pts[0][0], p[1] - pts[0][1]
        lateral = max(lateral, abs(-math.sin(h0) * dx + math.cos(h0) * dy))
    return dict(final_speed=final_speed, speed_delta=speed_delta, max_abs_jerk=max_abs_jerk,
                heading_change=heading_change, lateral_disp=lateral)


def zscore(values: list[float]) -> list[float]:
    m = _mean(values); s = statistics.pstdev(values) if len(values) > 1 else 0.0
    return [((v - m) / s if s > 1e-9 else 0.0) for v in values]


def score_aware(fz: dict[str, float], intents: set[str]) -> float:
    s = -1.0 * fz["max_abs_jerk"]                              # comfort, always
    if intents & STOP_LIKE:
        s += 1.5 * fz["speed_delta"] - 1.5 * fz["final_speed"]
    if "turn_left" in intents:
        s += 1.0 * fz["heading_change"]
    if "turn_right" in intents:
        s -= 1.0 * fz["heading_change"]
    if intents & {"lane_change_left", "lane_change_right", "avoid"}:
        s += 1.0 * fz["lateral_disp"]
    if not intents or (intents & {"go_straight", "proceed", "keep"}):
        s -= 0.5 * fz["lateral_disp"]
    return s


def score_blind(fz: dict[str, float]) -> float:
    return -1.0 * fz["max_abs_jerk"] - 0.5 * fz["lateral_disp"]


def select(cands: list[dict[str, Any]], scorer, gate_margin: float, first_idx: int):
    scores = [scorer(c["fz"]) for c in cands]
    best = max(range(len(cands)), key=lambda i: scores[i])
    if gate_margin > 0 and scores[best] - scores[first_idx] < gate_margin:
        best = first_idx
    return best, scores


def winloss(rer: list[float], base: list[float]):
    w = sum(1 for a, b in zip(rer, base) if a < b - 1e-9)
    l = sum(1 for a, b in zip(rer, base) if a > b + 1e-9)
    return {"win": w, "loss": l, "tie": len(rer) - w - l}


def bootstrap_improvement(clips, sel_key, n_boot, seed):
    """Paired bootstrap CI on mean (first_ade - sel_ade); positive = selector better than first-sample."""
    import numpy as np
    if not clips or n_boot <= 0:
        return {}
    d_ade = np.array([c["first_ade"] - c[sel_key + "_ade"] for c in clips], dtype=float)
    d_fde = np.array([c["first_fde"] - c[sel_key + "_fde"] for c in clips], dtype=float)
    rng = np.random.default_rng(seed); n = len(clips)

    def ci(d):
        idx = rng.integers(0, n, size=(n_boot, n))
        means = d[idx].mean(axis=1)
        return {"mean_improve_m": round(float(d.mean()), 4),
                "ci95": [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)],
                "frac_better": round(float((means > 0).mean()), 3)}

    return {"ade": ci(d_ade), "fde": ci(d_fde)}


def analyze(clips: list[dict[str, Any]]) -> dict[str, Any]:
    if not clips:
        return {"num_clips": 0}
    fa = [c["first_ade"] for c in clips]; ff = [c["first_fde"] for c in clips]
    aa = [c["aware_ade"] for c in clips]; af = [c["aware_fde"] for c in clips]
    ba = [c["blind_ade"] for c in clips]; bf = [c["blind_fde"] for c in clips]
    oa = [c["oracle_ade"] for c in clips]; of = [c["oracle_fde"] for c in clips]
    ra = [c["rand_ade"] for c in clips]
    ca = [c["centroid_ade"] for c in clips]; cf = [c["centroid_fde"] for c in clips]
    mca, mcf = _mean(ca), _mean(cf)
    mfa, moa, maa, mba = _mean(fa), _mean(oa), _mean(aa), _mean(ba)
    mff, mof, maf, mbf = _mean(ff), _mean(of), _mean(af), _mean(bf)

    def closed(m_re, m_first, m_oracle):
        denom = m_first - m_oracle
        return round((m_first - m_re) / denom * 100, 2) if abs(denom) > 1e-9 else None

    return {
        "num_clips": len(clips),
        "first_sample_ade": round(mfa, 4), "first_sample_fde": round(mff, 4),
        "rerank_aware_ade": round(maa, 4), "rerank_aware_fde": round(maf, 4),
        "rerank_blind_ade": round(mba, 4), "rerank_blind_fde": round(mbf, 4),
        "oracle_ade": round(moa, 4), "oracle_fde": round(mof, 4), "random_ade": round(_mean(ra), 4),
        "aware_gap_closed_ade_pct": closed(maa, mfa, moa),
        "aware_gap_closed_fde_pct": closed(maf, mff, mof),
        "blind_gap_closed_ade_pct": closed(mba, mfa, moa),
        "aware_intervention_rate": round(_mean([1.0 if c["aware_idx"] != c["first_idx"] else 0.0 for c in clips]), 3),
        "centroid_ade": round(mca, 4), "centroid_fde": round(mcf, 4),
        "centroid_gap_closed_ade_pct": closed(mca, mfa, moa),
        "centroid_vs_first": winloss(ca, fa),
        "aware_vs_first": winloss(aa, fa),
        "aware_vs_blind": winloss(aa, ba),
    }


def main() -> int:
    a = parse_args()
    import numpy as np
    bp = baseline_output_paths(split=a.split, run_name=a.run_name)
    preds = read_jsonl(a.predictions or bp["predictions"])
    traj = np.load(a.trajectories or bp["trajectories"])

    groups: dict[str, list[dict[str, Any]]] = {}
    for p in preds:
        pk, gk = p.get("pred_xyz_npz_key"), p.get("gt_xyz_npz_key")
        if pk not in traj or gk not in traj:
            continue
        pred_xyz = traj[pk].tolist()
        m = compute_trajectory_metrics(pred_xyz, traj[gk].tolist(), time_step=a.time_step)
        groups.setdefault(p["sample_id"], []).append({
            "tid": p.get("trajectory_sample_id", 0), "ade": m["ade_m"], "fde": m["fde_m"],
            "xyz": pred_xyz, "feat": candidate_features(pred_xyz, a.time_step),
            "cot": p.get("cot"), "meta_action": p.get("meta_action"), "answer": p.get("answer"),
        })

    clips: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for sid, cands in groups.items():
        cands.sort(key=lambda c: c["tid"])
        first_idx = next((i for i, c in enumerate(cands) if c["tid"] == 0), 0)
        c0 = cands[first_idx]
        intents = set(parse_intents({"cot": c0["cot"], "meta_action": c0["meta_action"], "answer": c0["answer"]}))
        for key in ("final_speed", "speed_delta", "max_abs_jerk", "heading_change", "lateral_disp"):
            zs = zscore([c["feat"][key] for c in cands])
            for c, z in zip(cands, zs):
                c.setdefault("fz", {})[key] = z
        aware_idx, aware_scores = select(cands, lambda fz: score_aware(fz, intents), a.gate_margin, first_idx)
        blind_idx, _ = select(cands, score_blind, a.gate_margin, first_idx)
        oracle_idx = min(range(len(cands)), key=lambda i: cands[i]["ade"])
        _arrs = [np.asarray(c["xyz"], dtype=float)[:, :2] for c in cands]
        _centroid = np.mean(_arrs, axis=0)
        centroid_idx = min(range(len(cands)), key=lambda i: float(np.mean(np.linalg.norm(_arrs[i] - _centroid, axis=1))))
        clips.append({
            "first_idx": first_idx, "aware_idx": aware_idx,
            "first_ade": c0["ade"], "first_fde": c0["fde"],
            "aware_ade": cands[aware_idx]["ade"], "aware_fde": cands[aware_idx]["fde"],
            "blind_ade": cands[blind_idx]["ade"], "blind_fde": cands[blind_idx]["fde"],
            "oracle_ade": cands[oracle_idx]["ade"], "oracle_fde": cands[oracle_idx]["fde"],
            "rand_ade": _mean([c["ade"] for c in cands]), "intents": sorted(intents),
            "centroid_ade": cands[centroid_idx]["ade"], "centroid_fde": cands[centroid_idx]["fde"],
        })
        selection_rows.append({
            "sample_id": sid, "intents": sorted(intents), "cot": c0["cot"],
            "first_tid": c0["tid"], "aware_tid": cands[aware_idx]["tid"], "oracle_tid": cands[oracle_idx]["tid"],
            "first_ade": round(c0["ade"], 3), "aware_ade": round(cands[aware_idx]["ade"], 3),
            "centroid_tid": cands[centroid_idx]["tid"], "centroid_ade": round(cands[centroid_idx]["ade"], 3),
            "oracle_ade": round(cands[oracle_idx]["ade"], 3),
            "aware_scores": [round(s, 3) for s in aware_scores],
        })

    sy = [c for c in clips if set(c["intents"]) & STOP_LIKE]
    sig_overall = {"centroid": bootstrap_improvement(clips, "centroid", a.bootstrap, a.seed),
                   "aware": bootstrap_improvement(clips, "aware", a.bootstrap, a.seed)}
    sig_sy = {"centroid": bootstrap_improvement(sy, "centroid", a.bootstrap, a.seed),
              "aware": bootstrap_improvement(sy, "aware", a.bootstrap, a.seed)} if sy else {}
    report = {"run_name": a.run_name, "split": a.split, "num_clips": len(clips),
              "candidates_per_clip": round(_mean([len(v) for v in groups.values()]), 2) if groups else 0,
              "gate_margin": a.gate_margin, "bootstrap": a.bootstrap,
              "overall": analyze(clips), "stop_yield_subset": analyze(sy),
              "significance_overall": sig_overall, "significance_stop_yield": sig_sy}
    print(json.dumps(report, indent=2))

    out = a.out or (Path("outputs/runs") / a.run_name / "analysis" / "rerank_report.json" if a.run_name else None)
    sel_out = a.selection_out or (Path("outputs/runs") / a.run_name / "analysis" / "rerank_selection.jsonl" if a.run_name else None)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[ok] wrote {out}")
    if sel_out:
        sel_out.parent.mkdir(parents=True, exist_ok=True)
        sel_out.write_text("\n".join(json.dumps(r) for r in selection_rows) + "\n", encoding="utf-8")
        print(f"[ok] wrote {sel_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
