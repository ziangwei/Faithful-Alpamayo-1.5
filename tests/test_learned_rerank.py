"""Synthetic-data tests for the Tier-2 learned reranker (scripts/06_learned_rerank.py).

The real numbers require the server-side baseline run (outputs/runs/<run>/), so here
we build a small controlled dataset where the candidate's *centrality* is exactly
what predicts quality and the deployed first sample is a deliberate bad outlier.
On such data we expect: MBR and the learned scorers both fix the outlier (beat
first-sample), the learned scorers land near MBR (no cheap-feature magic), and the
oracle lower-bounds everyone. This verifies the LOO-CV pipeline end-to-end without
a model or GPU.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


M6 = _load_module("learned_rerank06", "scripts/06_learned_rerank.py")


def _make_synthetic(num_clips: int = 24, num_cand: int = 5, horizon: int = 12, seed: int = 0):
    """Straight GT along +x; each candidate is GT shifted laterally by a constant y.

    ADE == |y offset| exactly. Candidate 0 (deployed) gets a large offset (outlier);
    the rest are small symmetric noise, so the most central candidate is good and the
    outlier is bad -- a clean stand-in for the real consensus signal.
    """
    rng = np.random.default_rng(seed)
    x = (0.5 * np.arange(horizon)).astype(float)
    gt = np.stack([x, np.zeros(horizon), np.zeros(horizon)], axis=1)

    preds: list[dict] = []
    traj: dict[str, np.ndarray] = {}
    for ci in range(num_clips):
        sid = f"clip{ci:03d}__0"
        traj[f"{sid}__gt_xyz"] = gt.copy()
        cot = "The ego vehicle should come to a full stop." if ci % 2 == 0 else "Proceed straight ahead."
        offsets = [2.5]  # candidate 0: outlier
        offsets += list(np.clip(rng.normal(0.0, 0.3, size=num_cand - 1), -0.9, 0.9))
        for tid, off in enumerate(offsets):
            key = f"{sid}__traj_{tid}__pred_xyz"
            traj[key] = np.stack([x, np.full(horizon, off), np.zeros(horizon)], axis=1)
            preds.append({
                "sample_id": sid, "trajectory_sample_id": tid,
                "pred_xyz_npz_key": key, "gt_xyz_npz_key": f"{sid}__gt_xyz",
                "cot": cot, "meta_action": "", "answer": "",
            })
    return preds, traj


class LearnedRerankTests(unittest.TestCase):
    def setUp(self):
        self.preds, self.traj = _make_synthetic()
        self.report, self.selection, self.clips = M6.build_report(
            self.preds, self.traj, time_step=0.1, n_boot=200, seed=1)
        self.ov = self.report["overall"]

    def test_pipeline_shapes_and_schema(self):
        self.assertEqual(self.report["num_clips"], 24)
        self.assertEqual(len(self.selection), 24)
        self.assertEqual(self.report["features"][-1], "dist_to_consensus")
        for key in ("verdict_logreg_vs_mbr", "verdict_ridge_vs_mbr",
                    "significance_overall", "stop_yield_subset"):
            self.assertIn(key, self.report)
        # every clip got a concrete LOO selection
        for c in self.clips:
            self.assertIn(c["logreg_idx"], range(len(c["ade"])))
            self.assertIn(c["ridge_idx"], range(len(c["ade"])))

    def test_selectors_beat_the_outlier_first_sample(self):
        self.assertGreater(self.ov["first_ade"], 2.0)          # deployed = bad outlier
        self.assertLess(self.ov["mbr_ade"], self.ov["first_ade"])
        self.assertLess(self.ov["logreg_ade"], self.ov["first_ade"])
        self.assertLess(self.ov["ridge_ade"], self.ov["first_ade"])

    def test_oracle_lower_bounds_everyone(self):
        o = self.ov["oracle_ade"]
        for k in ("first_ade", "mbr_ade", "ridge_ade", "logreg_ade", "random_ade"):
            self.assertLessEqual(o, self.ov[k] + 1e-6)

    def test_learned_is_near_mbr_not_magically_better(self):
        # cheap features carry no signal beyond centrality -> learned ~= MBR
        self.assertLess(abs(self.ov["logreg_ade"] - self.ov["mbr_ade"]), 0.5)
        self.assertLess(abs(self.ov["ridge_ade"] - self.ov["mbr_ade"]), 0.5)

    def test_stop_yield_subset_is_populated(self):
        self.assertGreater(self.report["stop_yield_subset"]["num_clips"], 0)

    def test_bootstrap_vs_mbr_has_ci_and_frac(self):
        ci = self.report["significance_overall"]["logreg_vs_mbr"]["ade"]
        self.assertEqual(len(ci["ci95"]), 2)
        self.assertGreaterEqual(ci["frac_better"], 0.0)
        self.assertLessEqual(ci["frac_better"], 1.0)


if __name__ == "__main__":
    unittest.main()
