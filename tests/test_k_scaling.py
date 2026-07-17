"""Tests for the k-of-N scaling analysis (scripts/03d_k_scaling.py).

The whole script rests on one claim: because the candidates are i.i.d. diffusion
samples, averaging over ALL C(N,k) subsets is the exact expectation of "what if we
had only drawn k samples". So the tests pin the sanity anchors that make the curve
trustworthy when it is later run on the server-side baseline:

  - k=1  : oracle-of-1 == MBR-of-1 == the random-candidate baseline (mean over cands).
  - k=N  : oracle-of-N == brute-force best-of-N; MBR-of-N == full-set consensus (04's rule).
  - oracle-of-k is monotone non-increasing in k, and its marginal gain is non-negative.
  - k=N oracle saturation == 100% (reaches the full-N ceiling by construction).

Real numbers require the server-side baseline run; here we use controlled synthetic data.
"""
from __future__ import annotations

import importlib.util
import itertools
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


M = _load("k_scaling03d", "scripts/03d_k_scaling.py")


def _make_synthetic(num_clips=40, num_cand=5, horizon=12, seed=0):
    """GT goes straight; each candidate = GT + a lateral ramp (so candidates are
    distinguishable and have genuinely different ADEs). cand 0 is a bad outlier on
    even clips, so first-sample is beatable. Half the clips carry a stop intent."""
    rng = np.random.default_rng(seed)
    x = (0.5 * np.arange(horizon)).astype(float)
    ramp = np.arange(horizon) / (horizon - 1)
    gt = np.stack([x, np.zeros(horizon), np.zeros(horizon)], axis=1)
    preds, traj = [], {}
    for ci in range(num_clips):
        sid = f"clip{ci:03d}"
        traj[f"{sid}__gt"] = gt.copy()
        cot = "The ego vehicle should come to a full stop." if ci % 2 == 0 else "Proceed straight."
        drifts = [2.5] + list(rng.normal(0, 0.5, num_cand - 1))
        for tid, dr in enumerate(drifts):
            key = f"{sid}__c{tid}"
            traj[key] = np.stack([x, float(dr) * ramp, np.zeros(horizon)], axis=1)
            preds.append({"sample_id": sid, "trajectory_sample_id": tid,
                          "pred_xyz_npz_key": key, "gt_xyz_npz_key": f"{sid}__gt",
                          "cot": cot, "meta_action": "", "answer": ""})
    return preds, traj


class PerClipCurves(unittest.TestCase):
    def test_k1_equals_random_and_kN_equals_oracle(self):
        # Three candidates with known ADEs; xy chosen so we can check the centroid rule.
        cands = [{"ade": 3.0, "fde": 6.0}, {"ade": 1.0, "fde": 2.0}, {"ade": 2.0, "fde": 4.0}]
        arrs = [np.array([[0.0, 0.0], [0.0, 3.0]]),   # far from others
                np.array([[0.0, 0.0], [0.0, 0.0]]),
                np.array([[0.0, 0.0], [0.0, 0.1]])]
        out = M.per_clip_curves(cands, arrs)
        # k=1: average over singletons -> mean ADE; oracle == mbr (subset of one picks itself)
        self.assertAlmostEqual(out[1]["oracle_ade"], 2.0)
        self.assertAlmostEqual(out[1]["mbr_ade"], 2.0)
        # k=N: single subset -> oracle is the true min ADE
        self.assertAlmostEqual(out[3]["oracle_ade"], 1.0)
        # oracle is monotone non-increasing in k
        self.assertGreaterEqual(out[1]["oracle_ade"] + 1e-9, out[2]["oracle_ade"])
        self.assertGreaterEqual(out[2]["oracle_ade"] + 1e-9, out[3]["oracle_ade"])

    def test_oracle_matches_bruteforce_average(self):
        rng = np.random.default_rng(1)
        n = 5
        cands = [{"ade": float(a), "fde": float(a) * 2} for a in rng.uniform(0.5, 4.0, n)]
        arrs = [rng.normal(0, 1, (6, 2)) for _ in range(n)]
        out = M.per_clip_curves(cands, arrs)
        for k in range(1, n + 1):
            subs = list(itertools.combinations(range(n), k))
            exp_oracle = np.mean([min(cands[i]["ade"] for i in s) for s in subs])
            self.assertAlmostEqual(out[k]["oracle_ade"], exp_oracle, places=9)

    def test_mbr_matches_bruteforce_centroid(self):
        rng = np.random.default_rng(2)
        n = 5
        cands = [{"ade": float(a), "fde": float(a) * 2} for a in rng.uniform(0.5, 4.0, n)]
        arrs = [rng.normal(0, 1, (6, 2)) for _ in range(n)]
        out = M.per_clip_curves(cands, arrs)
        for k in range(1, n + 1):
            subs = list(itertools.combinations(range(n), k))
            picks = []
            for s in subs:
                centroid = np.mean([arrs[i] for i in s], axis=0)
                sel = min(s, key=lambda i: float(np.mean(np.linalg.norm(arrs[i] - centroid, axis=1))))
                picks.append(cands[sel]["ade"])
            self.assertAlmostEqual(out[k]["mbr_ade"], float(np.mean(picks)), places=9)


class BuildReport(unittest.TestCase):
    def setUp(self):
        preds, traj = _make_synthetic()
        self.report, self.clips = M.build_report(preds, traj, bootstrap=200, seed=0)
        self.ov = self.report["overall"]
        self.per_k = self.ov["per_k"]

    def test_schema(self):
        self.assertEqual(self.report["num_candidates"], 5)
        self.assertEqual(self.report["num_clips"], 40)
        self.assertEqual(self.report["dropped_incomplete_clips"], 0)
        self.assertGreater(self.report["stop_yield_subset"]["num_clips"], 0)
        self.assertEqual(len(self.per_k), 5)
        for r in self.per_k:
            for key in ("oracle_ade", "mbr_ade", "oracle_sat_pct",
                        "mbr_gap_closed_ade_pct", "mbr_vs_first_ade"):
                self.assertIn(key, r)

    def test_k1_anchor_equals_random_baseline(self):
        # k=1 MBR == the reported random-candidate baseline (mean over all candidates).
        self.assertAlmostEqual(self.per_k[0]["mbr_ade"], self.report["random_candidate_ade"], places=4)
        self.assertAlmostEqual(self.per_k[0]["oracle_ade"], self.per_k[0]["mbr_ade"], places=6)
        # k=1 has no predecessor -> marginals are None
        self.assertIsNone(self.per_k[0]["oracle_marginal_ade"])
        self.assertIsNone(self.per_k[0]["mbr_marginal_ade"])

    def test_kN_oracle_matches_independent_bestof(self):
        # Recompute best-of-5 mean ADE straight from the synthetic data.
        preds, traj = _make_synthetic()
        by_clip: dict[str, list[float]] = {}
        gt = None
        for p in preds:
            pk, gk = p["pred_xyz_npz_key"], p["gt_xyz_npz_key"]
            pred, g = np.asarray(traj[pk]), np.asarray(traj[gk])
            ade = float(np.mean(np.linalg.norm(pred[:, :2] - g[:, :2], axis=1)))
            by_clip.setdefault(p["sample_id"], []).append(ade)
        exp = float(np.mean([min(v) for v in by_clip.values()]))
        self.assertAlmostEqual(self.per_k[-1]["oracle_ade"], exp, places=4)
        self.assertAlmostEqual(self.per_k[-1]["oracle_ade"], self.ov["oracle_full_n_ade"], places=6)

    def test_oracle_monotone_and_saturation(self):
        oracle = [r["oracle_ade"] for r in self.per_k]
        for a, b in zip(oracle, oracle[1:]):
            self.assertGreaterEqual(a + 1e-9, b)                    # non-increasing
        for r in self.per_k[1:]:
            self.assertGreaterEqual(r["oracle_marginal_ade"], -1e-9)  # marginal gain >= 0
        self.assertAlmostEqual(self.per_k[-1]["oracle_sat_pct"], 100.0, places=2)
        self.assertLessEqual(self.per_k[0]["oracle_sat_pct"], self.per_k[-1]["oracle_sat_pct"] + 1e-9)

    def test_oracle_lower_bounds_everything(self):
        for r in self.per_k:
            self.assertLessEqual(self.ov["oracle_full_n_ade"], r["oracle_ade"] + 1e-6)
            self.assertLessEqual(r["oracle_ade"], r["mbr_ade"] + 1e-6)


if __name__ == "__main__":
    unittest.main()
