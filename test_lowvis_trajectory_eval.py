#!/usr/bin/env python3

import unittest

import numpy as np

from evaluate_lowvis_trajectory_diffusion import (
    classification_metrics,
    energy_score,
    fit_thresholds,
    predict_classes,
    reliability_table,
    target_classes,
    variogram_score,
)


class TrajectoryEvaluationTests(unittest.TestCase):
    def test_threshold_and_class_metrics(self):
        visibility = np.array([[200.0, 700.0, 3000.0], [3000.0, 3000.0, 300.0]], dtype=np.float32)
        probs = np.array(
            [
                [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]],
                [[0.05, 0.05, 0.9], [0.05, 0.05, 0.9], [0.8, 0.1, 0.1]],
            ],
            dtype=np.float32,
        )
        mask = np.ones_like(visibility, dtype=bool)
        thresholds = fit_thresholds(probs, visibility, mask)
        pred = predict_classes(probs, thresholds["fog_threshold"], thresholds["mist_threshold"])
        metrics = classification_metrics(target_classes(visibility), pred)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        rel = reliability_table(probs, target_classes(visibility), mask)
        self.assertEqual(set(rel["class"]), {"Fog", "Mist", "Low-vis"})

    def test_joint_scores_are_zero_for_perfect_ensemble(self):
        target = np.arange(37, dtype=np.float32)[None, :]
        samples = np.repeat(target[:, None, :], 3, axis=1)
        np.testing.assert_allclose(energy_score(samples, target), 0.0, atol=1e-8)
        np.testing.assert_allclose(variogram_score(samples, target), 0.0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()

