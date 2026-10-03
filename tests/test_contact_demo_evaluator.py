"""Tests for repeatable contact-task evaluation and outcome aggregation."""

import unittest

import numpy as np

from evaluation.contact_demo_evaluator import aggregate_contact_runs, evaluate_contact_trajectory
from perception.task_layout import TaskLayout


class ContactDemoEvaluatorTests(unittest.TestCase):
    def test_fresh_runs_are_deterministic_and_keep_human_label_separate(self):
        layout = TaskLayout((0.5, 0.8), (0.5, 0.3))
        mapping = (0.55, 0.75, -0.22, 0.22)
        pickup = np.array((0.65, 0.132))
        dropoff = np.array((0.65, -0.088))
        trajectory = np.array([
            [0.0, 0.65, 0.132, 0.62],
            [0.2, 0.65, 0.132, 0.62],
            [0.4, 0.65, 0.00, 0.62],
            [0.6, 0.65, -0.088, 0.62],
        ])
        first = evaluate_contact_trajectory(trajectory, pickup, dropoff, human_label="riusciti")
        second = evaluate_contact_trajectory(trajectory, pickup, dropoff, human_label="riusciti")
        self.assertEqual(first["robot_success"], second["robot_success"])
        self.assertEqual(first["failure_reason"], second["failure_reason"])
        np.testing.assert_allclose(first["transitions"][0]["state"], second["transitions"][0]["state"])
        self.assertEqual(first["human_label"], "riusciti")
        self.assertIn("robot_success", first)
        self.assertNotEqual(first["human_label"], first["robot_success"])

    def test_aggregate_denominator_includes_trigger_failures(self):
        runs = [
            {"human_label": "riusciti", "robot_success": True, "failure_reason": None},
            {"human_label": "riusciti", "robot_success": False, "failure_reason": "pickup_trigger_not_reached"},
            {"human_label": "falliti", "robot_success": False, "failure_reason": "dropoff_trigger_not_reached"},
        ]
        aggregate = aggregate_contact_runs(runs)
        self.assertEqual(aggregate["episodes"], 3)
        self.assertEqual(aggregate["human_label_counts"], {"riusciti": 2, "falliti": 1})
        self.assertEqual(aggregate["by_human_label"]["riusciti"]["robot_failures"], 1)
        self.assertEqual(aggregate["confusion"]["human_success_robot_failure"], 1)
        self.assertEqual(aggregate["failure_reasons"]["pickup_trigger_not_reached"], 1)


if __name__ == "__main__":
    unittest.main()
