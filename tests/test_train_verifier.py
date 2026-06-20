"""Tests for the v2.0 verifier head (scripts/08_train_verifier.py).

We check (1) the MLP backprop is correct (numerical gradient check), and (2) the
geom-vs-(geom+scene) ABLATION behaves correctly on controlled synthetic data:
  - scene-informative: the scene vector tells which candidate matches GT (geometry
    alone cannot), so geom+scene beats MBR and the scene marginal is significant.
  - scene-noise: a random scene vector adds nothing, so the marginal is NOT
    significantly positive (no false positive).
Real numbers require dumping the frozen-VLM hidden states on the server.
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


M8 = _load("verifier08", "scripts/08_train_verifier.py")

# 5 fixed candidate "signatures" (lateral drifts): index 0 is the bad outlier,
# 1..4 are distinct good shapes. GT each clip = the scene-preferred good shape, so
# which good candidate is best is decided by the scene, not by geometry.
DRIFTS = [3.0, -0.7, -0.25, 0.25, 0.7]


def _make_scene_data(informative: bool, num_clips=200, horizon=12, seed=0):
    rng = np.random.default_rng(seed)
    x = (0.5 * np.arange(horizon)).astype(float)
    ramp = np.arange(horizon) / (horizon - 1)
    preds, traj, scene = [], {}, {}
    for ci in range(num_clips):
        sid = f"clip{ci:04d}__0"
        pref = int(rng.integers(1, 5))                       # which good candidate matches GT
        traj[f"{sid}__gt_xyz"] = np.stack([x, DRIFTS[pref] * ramp, np.zeros(horizon)], axis=1)
        cot = "The ego vehicle should come to a full stop." if ci % 2 == 0 else "Proceed straight."
        for tid, dr in enumerate(DRIFTS):
            key = f"{sid}__traj_{tid}__pred_xyz"
            traj[key] = np.stack([x, dr * ramp, np.zeros(horizon)], axis=1)
            preds.append({"sample_id": sid, "trajectory_sample_id": tid, "pred_xyz_npz_key": key,
                          "gt_xyz_npz_key": f"{sid}__gt_xyz", "cot": cot, "meta_action": "", "answer": ""})
        if informative:
            v = np.zeros(4); v[pref - 1] = 1.0
            scene[sid] = v + rng.normal(0, 0.05, 4)
        else:
            scene[sid] = rng.normal(0, 1.0, 4)
    return preds, traj, scene


class GradCheck(unittest.TestCase):
    def test_backprop_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        model = M8.MLP(4, hidden=6, seed=1)
        X = rng.standard_normal((10, 4)); y = rng.standard_normal(10)

        def loss():
            return float(((model.forward(X) - y) ** 2).mean())

        s = model.forward(X, cache=True)
        g = model.backward((2.0 / len(y)) * (s - y))
        eps = 1e-5
        for name in ("W1", "b1", "w2"):
            flat = model.P[name].reshape(-1); ga = g[name].reshape(-1)
            for i in range(len(flat)):
                o = flat[i]; flat[i] = o + eps; lp = loss(); flat[i] = o - eps; lm = loss(); flat[i] = o
                self.assertLess(abs((lp - lm) / (2 * eps) - ga[i]), 1e-4, f"{name}[{i}]")
        o = model.P["b2"]; model.P["b2"] = o + eps; lp = loss(); model.P["b2"] = o - eps; lm = loss(); model.P["b2"] = o
        self.assertLess(abs((lp - lm) / (2 * eps) - g["b2"]), 1e-4)


class Ablation(unittest.TestCase):
    def test_informative_scene_beats_mbr_and_marginal_significant(self):
        preds, traj, scene = _make_scene_data(True, seed=1)
        rep, _ = M8.build_report(preds, traj, scene, hidden=32, steps=800, lr=0.02, l2=1e-3,
                                 kfolds=5, n_boot=300, seed=1)
        ov = rep["overall"]
        self.assertTrue(rep["scene_vectors_present"])
        self.assertLess(ov["geomscene_ade"], ov["mbr_ade"])                       # scene head beats MBR
        self.assertGreater(rep["significance_overall"]["geomscene_vs_geom"]["ade"]["ci95"][0], 0.0)  # marginal significant
        self.assertIn("ADDS", rep["verdict_scene_marginal"])
        self.assertLess(abs(ov["geom_ade"] - ov["mbr_ade"]), 0.4)                 # geom-only ~= MBR (reproduces B/C)

    def test_noise_scene_adds_no_significant_signal(self):
        preds, traj, scene = _make_scene_data(False, seed=2)
        rep, _ = M8.build_report(preds, traj, scene, hidden=32, steps=800, lr=0.02, l2=1e-3,
                                 kfolds=5, n_boot=300, seed=2)
        # the scene marginal must NOT be significantly positive (no false positive)
        self.assertLessEqual(rep["significance_overall"]["geomscene_vs_geom"]["ade"]["ci95"][0], 0.05)

    def test_no_scene_runs_geom_only(self):
        preds, traj, _ = _make_scene_data(True, num_clips=40, seed=3)
        rep, _ = M8.build_report(preds, traj, None, hidden=16, steps=200, n_boot=0, seed=3)
        self.assertFalse(rep["scene_vectors_present"])
        self.assertIn("geom_ade", rep["overall"])
        self.assertNotIn("geomscene_ade", rep["overall"])


if __name__ == "__main__":
    unittest.main()
