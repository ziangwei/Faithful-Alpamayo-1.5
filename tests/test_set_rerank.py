"""Tests for the set-aggregator reranker (scripts/07_set_rerank.py).

Two things matter here: (1) the hand-derived backprop is correct (we numerically
gradient-check it), and (2) the k-fold pipeline runs end-to-end and behaves sanely on
controlled synthetic data (beats the outlier first-sample; lands near MBR; oracle lower-
bounds everyone). Real numbers require the server-side baseline run.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


M7 = _load("set_rerank07", "scripts/07_set_rerank.py")


def _make_synthetic(num_clips=60, num_cand=5, horizon=12, seed=0):
    # GT goes straight; each candidate drifts laterally by a ramp that GROWS along the
    # path (so the divergence survives the "relative-to-start" waypoint encoding and the
    # candidates are actually distinguishable, like real samples). cand 0 = bad outlier.
    rng = np.random.default_rng(seed)
    x = (0.5 * np.arange(horizon)).astype(float)
    ramp = np.arange(horizon) / (horizon - 1)            # 0..1
    gt = np.stack([x, np.zeros(horizon), np.zeros(horizon)], axis=1)
    preds, traj = [], {}
    for ci in range(num_clips):
        sid = f"clip{ci:03d}__0"
        traj[f"{sid}__gt_xyz"] = gt.copy()
        cot = "The ego vehicle should come to a full stop." if ci % 2 == 0 else "Proceed straight."
        drifts = [3.0] + list(np.clip(rng.normal(0, 0.4, num_cand - 1), -1.0, 1.0))
        for tid, dr in enumerate(drifts):
            key = f"{sid}__traj_{tid}__pred_xyz"
            traj[key] = np.stack([x, dr * ramp, np.zeros(horizon)], axis=1)
            preds.append({"sample_id": sid, "trajectory_sample_id": tid,
                          "pred_xyz_npz_key": key, "gt_xyz_npz_key": f"{sid}__gt_xyz",
                          "cot": cot, "meta_action": "", "answer": ""})
    return preds, traj


class GradCheck(unittest.TestCase):
    def test_backprop_matches_finite_differences(self):
        rng = np.random.default_rng(1)
        C, N, D, H = 3, 5, 4, 5
        model = M7.SetRegressor(D, hidden=H, seed=2)
        X = rng.standard_normal((C, N, D))
        T = rng.standard_normal((C, N))

        def loss():
            s = model.forward(X)
            return float(((s - T) ** 2).mean())

        s = model.forward(X, cache=True)
        dS = (2.0 / T.size) * (s - T)
        g = model.backward(dS)

        eps = 1e-5
        for name in ("W1", "b1", "W2", "b2", "w3"):
            P = model.P[name]
            flat = P.reshape(-1)
            ga = g[name].reshape(-1)
            for i in range(len(flat)):
                orig = flat[i]
                flat[i] = orig + eps; lp = loss()
                flat[i] = orig - eps; lm = loss()
                flat[i] = orig
                num = (lp - lm) / (2 * eps)
                self.assertLess(abs(num - ga[i]), 1e-4,
                                f"{name}[{i}] analytic {ga[i]:.6f} vs numeric {num:.6f}")
        # scalar bias b3
        orig = model.P["b3"]
        model.P["b3"] = orig + eps; lp = loss()
        model.P["b3"] = orig - eps; lm = loss()
        model.P["b3"] = orig
        self.assertLess(abs((lp - lm) / (2 * eps) - g["b3"]), 1e-4)

    def test_fit_reduces_loss(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((20, 5, 4))
        T = rng.standard_normal((20, 5))
        model = M7.SetRegressor(4, hidden=8, seed=0)
        before = model.loss(X, T)
        model.fit(X, T, steps=300, lr=0.02, seed=0)
        self.assertLess(model.loss(X, T), before)


class Pipeline(unittest.TestCase):
    def setUp(self):
        preds, traj = _make_synthetic()
        self.report, self.sel, self.clips = M7.build_report(
            preds, traj, hidden=16, steps=600, lr=0.03, kfolds=5, n_boot=200, seed=1)
        self.ov = self.report["overall"]

    def test_schema_and_shapes(self):
        self.assertEqual(self.report["num_clips"], 60)
        self.assertEqual(len(self.sel), 60)
        self.assertIn("verdict_setnn_vs_mbr", self.report)
        self.assertGreater(self.report["stop_yield_subset"]["num_clips"], 0)

    def test_beats_outlier_first_and_near_mbr(self):
        self.assertGreater(self.ov["first_ade"], 1.0)
        self.assertLess(self.ov["setnn_ade"], self.ov["first_ade"] - 0.3)   # clearly beats deployed outlier
        self.assertLess(self.ov["setnn_ade"], self.ov["mbr_ade"] + 0.5)     # in MBR's ballpark

    def test_oracle_lower_bounds(self):
        o = self.ov["oracle_ade"]
        for k in ("first_ade", "mbr_ade", "setnn_ade", "random_ade"):
            self.assertLessEqual(o, self.ov[k] + 1e-6)


if __name__ == "__main__":
    unittest.main()
