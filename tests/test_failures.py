import unittest

from faithful_vla.failures import (
    build_case_report_markdown,
    rank_failure_rows,
    summarize_failures,
)


class FailureMiningTests(unittest.TestCase):
    def test_ranks_inconsistent_cases_before_plain_high_error_cases(self):
        rows = [
            {
                "clip_id": "consistent-high-fde",
                "sample_id": "a",
                "ade_m": 5.0,
                "fde_m": 20.0,
                "is_consistent": True,
                "failed_checks": [],
                "cot": "Proceed through the lane.",
                "intents": ["go_straight"],
            },
            {
                "clip_id": "stop-mismatch",
                "sample_id": "b",
                "ade_m": 1.0,
                "fde_m": 2.0,
                "is_consistent": False,
                "failed_checks": ["stop"],
                "cot": "Stop for the blocked lane.",
                "intents": ["stop"],
            },
        ]

        ranked = rank_failure_rows(rows, top_k=2)

        self.assertEqual(ranked[0]["clip_id"], "stop-mismatch")
        self.assertEqual(ranked[0]["failure_rank"], 1)
        self.assertGreater(ranked[0]["failure_score"], ranked[1]["failure_score"])

    def test_summarizes_failure_rows(self):
        rows = [
            {
                "clip_id": "a",
                "ade_m": 1.0,
                "fde_m": 2.0,
                "is_consistent": False,
                "failed_checks": ["stop"],
            },
            {
                "clip_id": "b",
                "ade_m": 3.0,
                "fde_m": 8.0,
                "is_consistent": False,
                "failed_checks": ["yield"],
            },
            {
                "clip_id": "c",
                "ade_m": 2.0,
                "fde_m": 4.0,
                "is_consistent": True,
                "failed_checks": [],
            },
        ]

        summary = summarize_failures(rows, top_rows=rank_failure_rows(rows, top_k=2))

        self.assertEqual(summary["num_samples"], 3)
        self.assertEqual(summary["num_inconsistent"], 2)
        self.assertAlmostEqual(summary["inconsistency_rate"], 2 / 3)
        self.assertEqual(summary["failed_check_counts"], {"stop": 1, "yield": 1})
        self.assertEqual(summary["top_failure_clip_ids"], ["b", "a"])

    def test_builds_case_report_markdown(self):
        summary = {
            "run_name": "alpamayo15_val_baseline_60",
            "num_samples": 60,
            "num_inconsistent": 9,
            "inconsistency_rate": 0.15,
            "failed_check_counts": {"stop": 8, "yield": 2},
        }
        rows = [
            {
                "failure_rank": 1,
                "clip_id": "clip-1",
                "sample_id": "clip-1__5100000__0",
                "failure_score": 123.5,
                "ade_m": 2.0,
                "fde_m": 9.0,
                "is_consistent": False,
                "failed_checks": ["stop"],
                "intents": ["stop"],
                "cot": "Stop because construction blocks the lane.",
                "meta_action": "",
            }
        ]

        markdown = build_case_report_markdown(summary, rows)

        self.assertIn("# Failure Mining Report", markdown)
        self.assertIn("alpamayo15_val_baseline_60", markdown)
        self.assertIn("clip-1", markdown)
        self.assertIn("Stop because construction blocks the lane.", markdown)


if __name__ == "__main__":
    unittest.main()
