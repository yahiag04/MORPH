"""Tests for hand-tracking and Cartesian trajectory evaluation metrics."""

import unittest

import numpy as np

from evaluation.demo_metrics import tracking_metrics, trajectory_metrics


class DemoMetricsTests(unittest.TestCase):
    def test_tracking_metrics_report_coverage_and_interpolated_frames(self):
        metrics = tracking_metrics(
            np.asarray((0.0, 0.1, 0.2, 0.3)),
            np.asarray((0.9, 0.0, 0.4, 0.8)),
            threshold=0.5,
        )
        self.assertEqual(metrics["sample_count"], 4)
        self.assertEqual(metrics["detected_frames"], 2)
        self.assertEqual(metrics["interpolated_frames"], 2)
        self.assertEqual(metrics["coverage_fraction"], 0.5)

    def test_zero_confidence_missing_frame_stays_missing_at_zero_threshold(self):
        metrics = tracking_metrics(np.asarray((0.0, 0.1)), np.asarray((0.0, 0.8)), threshold=0.0)
        self.assertEqual(metrics["coverage_fraction"], 0.5)
        self.assertEqual(metrics["interpolated_frames"], 1)

    def test_straight_path_has_known_length_and_zero_jerk(self):
        times = np.asarray((0.0, 0.1, 0.2, 0.3))
        points = np.column_stack((2 * times, np.zeros((4, 2))))
        metrics = trajectory_metrics(times, points)
        self.assertAlmostEqual(metrics["path_length_m"], 0.6, places=8)
        self.assertLess(metrics["jerk_rms_mps3"], 1e-8)

    def test_stationary_path_has_zero_length_and_zero_jerk(self):
        metrics = trajectory_metrics(np.asarray((0.0, 0.2)), np.ones((2, 3)))
        self.assertEqual(metrics["path_length_m"], 0.0)
        self.assertEqual(metrics["jerk_rms_mps3"], 0.0)

    def test_nonstationary_short_path_has_undefined_jerk(self):
        metrics = trajectory_metrics(np.asarray((0.0, 0.1, 0.35)), np.asarray(((0, 0, 0), (1, 0, 0), (2, 0, 0))))
        self.assertIsNone(metrics["jerk_rms_mps3"])

    def test_rejects_non_increasing_timestamps(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            trajectory_metrics(np.asarray((0.0, 0.1, 0.1, 0.2)), np.zeros((4, 3)))


if __name__ == "__main__":
    unittest.main()
