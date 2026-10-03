"""Tests for deterministic headless Panda trajectory evaluation."""

from unittest import mock
import unittest

import numpy as np

from evaluation.demo_evaluator import evaluate_trajectory


class DemoEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.trajectory = np.asarray(((0.386977878, 0.0, 0.854487017), (0.406977878, 0.0, 0.854487017)))

    def test_repeated_runs_start_from_the_same_seed_and_match(self):
        first_log, first_metrics = evaluate_trajectory(self.trajectory)
        second_log, second_metrics = evaluate_trajectory(self.trajectory)
        np.testing.assert_allclose(first_log.target_xyz, second_log.target_xyz)
        np.testing.assert_allclose(first_log.actual_xyz, second_log.actual_xyz, atol=1e-10)
        self.assertEqual(first_metrics, second_metrics)
        self.assertEqual(first_metrics["ik_nonconverged_count"], 0)
        self.assertGreater(first_metrics["sample_count"], 0)

    def test_reports_ik_nonconvergence_in_metrics(self):
        failed_result = mock.Mock(q=np.asarray((0.0, -0.5, 0.0, -1.7, 0.0, 2.0, 0.0)), converged=False)
        with mock.patch("simulation.trajectory_player.solve_position_ik", return_value=failed_result):
            _log, metrics = evaluate_trajectory(self.trajectory)
        self.assertGreater(metrics["ik_nonconverged_count"], 0)


if __name__ == "__main__":
    unittest.main()
