import unittest
from pathlib import Path

from faithful_vla.run_migration import build_migration_specs


class RunMigrationTests(unittest.TestCase):
    def test_maps_flat_run_files_into_nested_directories(self):
        specs = build_migration_specs(
            run_name="alpamayo15_val_baseline_60",
            split="val",
        )
        by_name = {spec.target.name: spec for spec in specs}

        self.assertEqual(
            by_name["val_predictions.jsonl"].target,
            Path(
                "outputs/runs/alpamayo15_val_baseline_60/"
                "baseline/val_predictions.jsonl"
            ),
        )
        self.assertEqual(
            by_name["summary.json"].target,
            Path("outputs/runs/alpamayo15_val_baseline_60/metrics/summary.json"),
        )

    def test_prefers_flat_run_source_before_legacy_global_source(self):
        specs = build_migration_specs(
            run_name="alpamayo15_val_baseline_60",
            split="val",
        )
        predictions = next(spec for spec in specs if spec.target.name == "val_predictions.jsonl")

        self.assertEqual(
            predictions.sources[0],
            Path("outputs/runs/alpamayo15_val_baseline_60/val_predictions.jsonl"),
        )
        self.assertEqual(
            predictions.sources[1],
            Path("outputs/baseline/val_predictions.jsonl"),
        )

    def test_metrics_sources_fall_back_to_legacy_metrics_directory(self):
        specs = build_migration_specs(
            run_name="alpamayo15_val_baseline_60",
            split="val",
        )
        summary = next(spec for spec in specs if spec.target.name == "summary.json")

        self.assertEqual(
            summary.sources,
            (
                Path("outputs/runs/alpamayo15_val_baseline_60/summary.json"),
                Path("outputs/metrics/summary.json"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
