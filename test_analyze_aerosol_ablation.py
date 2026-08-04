import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import analyze_aerosol_ablation as aerosol


class AerosolAblationAnalysisTests(unittest.TestCase):
    def test_legacy_full_layout_is_width_audited_without_config(self):
        width = 12 * len(aerosol.LEGACY_FULL_DYNAMIC_ORDER) + 6 + 36
        fake_matrix = type("FakeMatrix", (), {"shape": (2, width)})()
        with patch.object(
            Path,
            "is_file",
            autospec=True,
            side_effect=lambda path: path.name == "X_val.npy",
        ), patch.object(aerosol.np, "load", return_value=fake_matrix):
            data_dir = Path("legacy_full_s2")
            window, dyn, order, cfg = aerosol.dynamic_layout(data_dir)
            self.assertEqual(window, 12)
            self.assertEqual(dyn, 27)
            self.assertEqual(order[-2:], ["PM10", "PM2P5"])
            self.assertEqual(cfg["dynamic_feature_order_source"], "legacy_full_27_width_audit")

    def test_average_precision_groups_tied_scores(self):
        y = np.array([1, 0, 1, 0], dtype=bool)
        score = np.array([0.8, 0.8, 0.4, 0.1])
        self.assertAlmostEqual(aerosol.average_precision(y, score), 7.0 / 12.0)

    def test_threshold_selection_respects_validation_fpr_cap(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
        score = np.array([0.95, 0.80, 0.45, 0.90, 0.40, 0.30, 0.20, 0.10])
        threshold, metrics = aerosol.select_threshold_at_fpr(y, score, target_fpr=0.20)
        self.assertAlmostEqual(threshold, 0.45)
        self.assertLessEqual(metrics["fpr"], 0.20)
        self.assertAlmostEqual(metrics["recall"], 1.0)

    def test_full_minus_no_pm_bootstrap_uses_paired_date_blocks(self):
        dates = pd.to_datetime(["2025-01-01", "2025-01-02"])
        full = pd.DataFrame(
            {"tp": [8, 9], "fp": [2, 1], "fn": [2, 1], "tn": [88, 89]},
            index=dates,
        )
        no_pm = pd.DataFrame(
            {"tp": [6, 7], "fp": [2, 1], "fn": [4, 3], "tn": [88, 89]},
            index=dates,
        )
        result = aerosol.bootstrap_metric_differences(full, no_pm, iterations=100, seed=7)
        recall = result.set_index("metric").loc["recall"]
        self.assertGreater(recall["ci_low"], 0.0)
        self.assertEqual(int(recall["date_blocks"]), 2)

    def test_month_relative_percentile_does_not_depend_on_absolute_scale(self):
        reference = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        months = np.array([1, 1, 1, 2, 2, 2])
        values = np.array([2.0, 20.0])
        value_months = np.array([1, 2])
        rank = aerosol.reference_percentile(reference, months, values, value_months)
        scaled = aerosol.reference_percentile(
            reference * 1000.0,
            months,
            values * 1000.0,
            value_months,
        )
        np.testing.assert_allclose(rank, scaled)
        np.testing.assert_allclose(rank, np.array([2.0 / 3.0, 2.0 / 3.0]))

    def test_environment_signature_is_explicitly_descriptive(self):
        signature = aerosol.environment_signature(95.0, 0.9, 0.8, 0.0)
        self.assertEqual(signature, "humid aerosol-rich weak-ventilation environment")

    def test_event_summary_reports_full_minus_no_pm_skill(self):
        times = pd.Series(pd.date_range("2025-02-27 21:00:00", periods=3, freq="h"))
        y = np.array([1, 1, 0], dtype=bool)
        full = np.array([1, 1, 0], dtype=bool)
        no_pm = np.array([1, 0, 0], dtype=bool)
        result = aerosol.event_summary(
            times,
            [pd.Timestamp("2025-02-27 22:00:00")],
            1,
            y,
            full,
            no_pm,
            np.array([95.0, 96.0, 50.0]),
            np.array([0.8, 0.9, 0.2]),
            np.array([1.0, 2.0, 8.0]),
            np.zeros(3),
        )
        self.assertAlmostEqual(float(result.loc[0, "delta_recall"]), 0.5)
        self.assertIn("aerosol-rich", result.loc[0, "descriptive_environment_signature"])


if __name__ == "__main__":
    unittest.main()
