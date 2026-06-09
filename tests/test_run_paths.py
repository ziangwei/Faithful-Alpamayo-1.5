import unittest
from pathlib import Path

from faithful_vla.run_paths import baseline_output_paths, metrics_output_paths


class RunPathTests(unittest.TestCase):
    def test_baseline_paths_use_run_name_when_provided(self):
        paths = baseline_output_paths(split="val", run_name="alpamayo15_val_baseline_60")

        self.assertEqual(
            paths["predictions"],
            Path("outputs/runs/alpamayo15_val_baseline_60/baseline/val_predictions.jsonl"),
        )
        self.assertEqual(
            paths["trajectories"],
            Path("outputs/runs/alpamayo15_val_baseline_60/baseline/val_trajectories.npz"),
        )
        self.assertEqual(
            paths["runtime"],
            Path("outputs/runs/alpamayo15_val_baseline_60/baseline/val_runtime.jsonl"),
        )

    def test_metrics_paths_use_run_name_for_inputs_and_outputs(self):
        paths = metrics_output_paths(split="val", run_name="alpamayo15_val_baseline_60")

        self.assertEqual(
            paths["predictions"],
            Path("outputs/runs/alpamayo15_val_baseline_60/baseline/val_predictions.jsonl"),
        )
        self.assertEqual(
            paths["trajectories"],
            Path("outputs/runs/alpamayo15_val_baseline_60/baseline/val_trajectories.npz"),
        )
        self.assertEqual(
            paths["summary"],
            Path("outputs/runs/alpamayo15_val_baseline_60/metrics/summary.json"),
        )
        self.assertEqual(
            paths["per_sample"],
            Path("outputs/runs/alpamayo15_val_baseline_60/metrics/per_sample_metrics.jsonl"),
        )
        self.assertEqual(
            paths["inconsistency"],
            Path("outputs/runs/alpamayo15_val_baseline_60/metrics/inconsistency_examples.jsonl"),
        )

    def test_rejects_path_like_run_name(self):
        with self.assertRaises(ValueError):
            baseline_output_paths(split="val", run_name="../bad")


if __name__ == "__main__":
    unittest.main()
