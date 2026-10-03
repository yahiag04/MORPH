"""Tests for normalized task layouts and contact-driven replay."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from perception.task_layout import TaskLayout
from simulation.contact_manipulation_env import ContactManipulationEnv
from simulation.demonstration_manipulation import run_demonstration_manipulation
from simulation.panda_env import DEFAULT_MODEL_PATH
from perception.retargeting import WorkspaceMapping


class TaskLayoutTests(unittest.TestCase):
    def test_json_round_trip_and_workspace_mapping(self):
        layout = TaskLayout((0.25, 0.3), (0.75, 0.7))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.json"
            layout.save(path)
            self.assertEqual(TaskLayout.load(path), layout)
        mapping = WorkspaceMapping(robot_x_min=0.4, robot_x_max=0.8, robot_y_min=-0.2, robot_y_max=0.2)
        np.testing.assert_allclose(layout.to_robot_xy(mapping)[0], (0.5, -0.08))
        np.testing.assert_allclose(layout.to_robot_xy(mapping)[1], (0.7, 0.08))

    def test_rejects_points_outside_unit_square(self):
        with self.assertRaisesRegex(ValueError, "unit square"):
            TaskLayout((-0.1, 0.2), (0.8, 0.8))


class DemonstrationManipulationTests(unittest.TestCase):
    def test_path_triggers_grasp_then_release_without_object_teleportation(self):
        env = ContactManipulationEnv(
            pickup_xyz=(0.60, 0.00, 0.412),
            dropoff_xyz=(0.58, 0.12, 0.416),
        )
        # Sparse path enters pickup, carries while closed, then enters dropoff.
        path = np.array([
            [0.00, 0.60, 0.00, 0.62],
            [0.10, 0.60, 0.00, 0.62],
            [0.20, 0.59, 0.06, 0.62],
            [0.30, 0.58, 0.12, 0.62],
        ])
        transitions = []
        result = run_demonstration_manipulation(
            env, path, (0.60, 0.00), (0.58, 0.12),
            on_control_step=lambda item: transitions.append(item),
        )
        phases = [row["phase"] for row in transitions]
        self.assertIn("grasp", phases)
        self.assertIn("carry", phases)
        self.assertIn("release", phases)
        self.assertEqual(result["phase"], "complete")
        grasp_index = next(i for i, row in enumerate(transitions) if row["phase"] == "grasp")
        release_index = next(i for i, row in enumerate(transitions) if row["phase"] == "release")
        self.assertEqual(transitions[grasp_index]["gripper_command"], "close")
        self.assertTrue(all(row["gripper_command"] == "close" for row in transitions[grasp_index:release_index]))
        self.assertEqual(transitions[release_index]["gripper_command"], "open")
        self.assertTrue(all(row["ik_converged"] for row in transitions))
        self.assertGreaterEqual(result["max_lift_m"], 0.0)
        self.assertFalse(result["success"])
        self.assertGreaterEqual(env.get_package_position()[2], env.config.table_surface_z - 0.02)

    def test_missing_trigger_is_explicit_failure(self):
        env = ContactManipulationEnv(
            pickup_xyz=(0.60, 0.00, 0.412), dropoff_xyz=(0.58, 0.12, 0.416)
        )
        path = np.array([[0.0, 0.40, 0.20, 0.62], [0.1, 0.41, 0.20, 0.62]])
        result = run_demonstration_manipulation(env, path, (0.52, -0.12), (0.58, 0.12))
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_reason"], "pickup_trigger_not_reached")


if __name__ == "__main__":
    unittest.main()
