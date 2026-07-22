import unittest

import numpy as np
import pandas as pd

import analyze_lowvis_operational_tradeoffs as audit


class OperationalTradeoffAuditTests(unittest.TestCase):
    def test_binary_metrics_and_threshold_selection(self):
        truth = np.array([True, True, False, False])
        score = np.array([0.9, 0.6, 0.7, 0.1])
        metrics = audit.binary_metrics(truth, score >= 0.5)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 1)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)

        curve = audit.threshold_curve(truth, score, points=11)
        selected = audit.select_thresholds(curve, precision_targets=[0.75], ifs_fpr=0.5)
        self.assertIn("max_csi", set(selected["selection_rule"]))
        self.assertIn("precision_at_least_0.75", set(selected["selection_rule"]))
        self.assertIn("matched_ifs_fpr", set(selected["selection_rule"]))

    def test_calibration_summary(self):
        truth = np.array([0, 0, 1, 1], dtype=float)
        score = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
        table, summary = audit.calibration_table(truth, score, bins=5)
        self.assertEqual(int(table["count"].sum()), 4)
        self.assertAlmostEqual(summary["prevalence"], 0.5)
        self.assertLess(summary["brier"], 0.05)

    def test_event_boundary_censoring(self):
        frame = pd.DataFrame(
            {
                "hour_offset": [-2, -1, 0, 1, 2],
                "obs_low_vis_count": [5, 10, 20, 10, 5],
                "pmst_low_vis_count": [8, 12, 25, 15, 8],
                "ifs_low_vis_count": [0, 2, 4, 2, 0],
                "obs_fog_count": [0, 2, 8, 2, 0],
                "pmst_fog_count": [0, 4, 12, 4, 0],
                "ifs_fog_count": [0, 0, 1, 0, 0],
            }
        )
        rows = pd.DataFrame(audit.event_dynamics_frame(frame, "event_1", activity_fraction=0.2))
        lowvis_model = rows[(rows["event_class"] == "low_vis") & (rows["source"] == "model")].iloc[0]
        self.assertTrue(bool(lowvis_model["obs_onset_left_censored"]))
        self.assertTrue(bool(lowvis_model["pred_dissipation_right_censored"]))
        self.assertTrue(np.isnan(lowvis_model["onset_error_h"]))


if __name__ == "__main__":
    unittest.main()
