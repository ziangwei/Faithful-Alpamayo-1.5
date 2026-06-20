"""Unit tests for the pure-Python parts of 02's --dump-hidden hook.

The full inference needs the 10B model + a GPU, so it can't run here. But the two
model-agnostic pieces -- the HiddenCapture forward-hook pooling and the VLM submodule
resolver -- are plain Python and are tested directly with lightweight fakes.
"""
from __future__ import annotations

import importlib.util
import types
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


M2 = _load("baseline_inference02", "scripts/02_run_baseline_inference.py")


class FakeTensor:
    """Mimics the tiny slice of the torch.Tensor API HiddenCapture uses."""

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=float)

    def dim(self):
        return self.arr.ndim

    @property
    def shape(self):
        return self.arr.shape

    def detach(self):
        return self

    def float(self):
        return self

    def mean(self, dim):
        return FakeTensor(self.arr.mean(axis=dim))

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class HiddenCaptureTests(unittest.TestCase):
    def test_keeps_longest_pass_and_mean_pools(self):
        cap = M2.HiddenCapture()
        rng = np.random.default_rng(0)
        prefill = rng.standard_normal((1, 10, 4))            # the prefill = longest forward
        cap(None, None, types.SimpleNamespace(last_hidden_state=FakeTensor(np.ones((1, 3, 4)))))
        cap(None, None, types.SimpleNamespace(last_hidden_state=FakeTensor(prefill)))
        cap(None, None, types.SimpleNamespace(last_hidden_state=FakeTensor(np.zeros((1, 1, 4)))))  # decode step
        v = cap.take()
        self.assertEqual(v.shape, (4,))                      # pooled to hidden-size
        np.testing.assert_allclose(v, prefill.mean(axis=(0, 1)), rtol=1e-6)

    def test_take_resets_state(self):
        cap = M2.HiddenCapture()
        cap(None, None, types.SimpleNamespace(last_hidden_state=FakeTensor(np.ones((1, 5, 3)))))
        self.assertIsNotNone(cap.take())
        self.assertIsNone(cap.take())                        # reset after take

    def test_accepts_tuple_output(self):
        cap = M2.HiddenCapture()
        cap(None, None, (FakeTensor(np.ones((1, 6, 2))),))
        np.testing.assert_allclose(cap.take(), np.ones(2))


class ResolveModuleTests(unittest.TestCase):
    def test_prefers_default_text_decoder_path(self):
        model = _nest(["vlm", "model", "language_model"], types.SimpleNamespace(_marker="decoder"))
        mod, path = M2.resolve_vlm_module(model, None)
        self.assertEqual(path, "vlm.model.language_model")
        self.assertEqual(mod._marker, "decoder")

    def test_falls_back_when_default_missing(self):
        model = _nest(["vlm", "language_model"], types.SimpleNamespace(_marker="lm"))
        mod, path = M2.resolve_vlm_module(model, None)
        self.assertEqual(path, "vlm.language_model")

    def test_honors_explicit_override(self):
        model = _nest(["backbone", "decoder"], types.SimpleNamespace(_marker="x"))
        mod, path = M2.resolve_vlm_module(model, "backbone.decoder")
        self.assertEqual(path, "backbone.decoder")

    def test_raises_when_nothing_matches(self):
        with self.assertRaises(AttributeError):
            M2.resolve_vlm_module(types.SimpleNamespace(), None)


class ExpertCaptureTests(unittest.TestCase):
    def test_keeps_last_pass_and_pools_per_candidate(self):
        cap = M2.ExpertCapture()
        rng = np.random.default_rng(0)
        step1 = rng.standard_normal((5, 8, 4))       # 5 candidates, 8 traj tokens, H=4
        last = rng.standard_normal((5, 8, 4))         # the final denoising step
        cap(None, None, types.SimpleNamespace(last_hidden_state=FakeTensor(step1)))
        cap(None, None, types.SimpleNamespace(last_hidden_state=FakeTensor(last)))
        v = cap.take()
        self.assertEqual(v.shape, (5, 4))             # per-candidate (b*, H)
        np.testing.assert_allclose(v, last.mean(axis=1), rtol=1e-6)  # last call, pooled over tokens
        self.assertIsNone(cap.take())                 # reset


def _nest(parts, leaf):
    node = leaf
    for part in reversed(parts):
        node = types.SimpleNamespace(**{part: node})
    return node


if __name__ == "__main__":
    unittest.main()
