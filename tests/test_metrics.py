import unittest

from faithful_vla.metrics import (
    DEFAULT_CONSISTENCY_THRESHOLDS,
    check_reasoning_action_consistency,
    compute_trajectory_metrics,
    parse_intents,
    summarize_metric_rows,
)


class MetricsTests(unittest.TestCase):
    def test_computes_ade_fde_and_speed_features(self):
        pred_xyz = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
        gt_xyz = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]

        metrics = compute_trajectory_metrics(pred_xyz, gt_xyz, time_step=1.0)

        self.assertAlmostEqual(metrics["ade_m"], 1.0)
        self.assertAlmostEqual(metrics["fde_m"], 2.0)
        self.assertAlmostEqual(metrics["average_speed_mps"], 1.5)
        self.assertAlmostEqual(metrics["final_speed_mps"], 2.0)
        self.assertAlmostEqual(metrics["speed_delta_mps"], -1.0)
        self.assertAlmostEqual(metrics["lateral_displacement_m"], 0.0)

    def test_parses_intents_from_reasoning_fields(self):
        intents = parse_intents(
            {
                "cot": "The vehicle should slow down and yield to cross traffic.",
                "meta_action": "brake",
            }
        )

        self.assertEqual(intents, ["slow_down", "yield"])

    def test_flags_stop_reasoning_when_trajectory_keeps_moving(self):
        behavior = {
            "final_speed_mps": 2.0,
            "speed_delta_mps": 0.0,
            "lateral_displacement_m": 0.0,
        }

        result = check_reasoning_action_consistency(
            intents=["stop"],
            behavior=behavior,
            thresholds=DEFAULT_CONSISTENCY_THRESHOLDS,
        )

        self.assertFalse(result["is_consistent"])
        self.assertEqual(result["failed_checks"], ["stop"])

    def test_accepts_slow_down_when_speed_decreases_enough(self):
        behavior = {
            "final_speed_mps": 1.0,
            "speed_delta_mps": 2.0,
            "lateral_displacement_m": 0.0,
        }

        result = check_reasoning_action_consistency(
            intents=["slow_down"],
            behavior=behavior,
            thresholds=DEFAULT_CONSISTENCY_THRESHOLDS,
        )

        self.assertTrue(result["is_consistent"])
        self.assertEqual(result["failed_checks"], [])

    def test_summarizes_metric_rows(self):
        rows = [
            {"ade_m": 1.0, "fde_m": 2.0, "is_consistent": True, "failed_checks": []},
            {"ade_m": 3.0, "fde_m": 4.0, "is_consistent": False, "failed_checks": ["stop"]},
        ]

        summary = summarize_metric_rows(rows)

        self.assertEqual(summary["num_samples"], 2)
        self.assertAlmostEqual(summary["mean_ade_m"], 2.0)
        self.assertAlmostEqual(summary["mean_fde_m"], 3.0)
        self.assertAlmostEqual(summary["reasoning_action_consistency_rate"], 0.5)
        self.assertEqual(summary["inconsistency_count_by_type"], {"stop": 1})


if __name__ == "__main__":
    unittest.main()
