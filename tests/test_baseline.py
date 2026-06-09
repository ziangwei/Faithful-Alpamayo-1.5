import unittest
from pathlib import Path

from faithful_vla.baseline import (
    build_dry_run_summary,
    build_prediction_row,
    load_manifest_records,
    select_manifest_records,
)


class BaselineManifestTests(unittest.TestCase):
    def test_selects_val_records_with_limit_preserving_manifest_order(self):
        manifest_path = Path(__file__).resolve().parent / "fixtures" / "baseline_manifest.jsonl"
        loaded = load_manifest_records(manifest_path)
        selected = select_manifest_records(loaded, split="val", limit=1)

        self.assertEqual([record["clip_id"] for record in selected], ["val-1"])

    def test_dry_run_summary_reports_selected_records_without_downloads(self):
        records = [
            {"clip_id": "val-1", "split": "val", "t0_us": 5100000},
            {"clip_id": "val-2", "split": "val", "t0_us": 5100000},
        ]

        summary = build_dry_run_summary(records, split="val", execute=False)

        self.assertEqual(summary["split"], "val")
        self.assertEqual(summary["selected_records"], 2)
        self.assertFalse(summary["execute"])
        self.assertFalse(summary["model_load_performed"])
        self.assertFalse(summary["dataset_load_performed"])

    def test_prediction_row_has_stable_metadata_and_npz_keys(self):
        record = {
            "clip_id": "clip-a",
            "split": "val",
            "t0_us": 5100000,
            "required_cameras": ["CAMERA_FRONT_WIDE_120FOV"],
        }

        row = build_prediction_row(
            record=record,
            sample_index=0,
            trajectory_sample_id=0,
            cot="slow down for traffic",
            meta_action="slow_down",
            answer="",
            runtime_sec=12.5,
            max_cuda_memory_gb=31.25,
        )

        self.assertEqual(row["clip_id"], "clip-a")
        self.assertEqual(row["sample_id"], "clip-a__5100000__0")
        self.assertEqual(row["trajectory_npz_key"], "clip-a__5100000__0__traj_0")
        self.assertEqual(row["split"], "val")
        self.assertEqual(row["required_cameras"], ["CAMERA_FRONT_WIDE_120FOV"])


if __name__ == "__main__":
    unittest.main()
