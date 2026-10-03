"""Tests for normalized task layouts and contact-driven replay."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from perception.task_layout import TaskLayout, detect_task_layout
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


    def test_detects_card_and_bowl_centers_inside_calibrated_workspace(self):
        import cv2
        frame = np.full((400, 600, 3), 255, dtype=np.uint8)
        # Brown bowl and a small navy packet, both well inside the marker quad.
        cv2.circle(frame, (420, 120), 38, (35, 90, 145), -1)
        cv2.rectangle(frame, (160, 250), (205, 315), (100, 45, 20), -1)
        corners = np.asarray(((0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)))
        layout = detect_task_layout(frame, corners)
        np.testing.assert_allclose(layout.pickup_uv, (0.28, 0.73), atol=0.04)
        np.testing.assert_allclose(layout.dropoff_uv, (0.72, 0.28), atol=0.04)

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
        grasp_index = next(i for i, row in enumerate(transitions) if row["phase"] == "grasp" and row["gripper_command"] == "close")
        release_index = next(i for i, row in enumerate(transitions) if row["phase"] == "release")
        self.assertEqual(transitions[grasp_index]["gripper_command"], "close")
        self.assertTrue(all(row["gripper_command"] == "close" for row in transitions[grasp_index:release_index]))
        self.assertEqual(transitions[release_index]["gripper_command"], "open")
        self.assertTrue(all(row["ik_converged"] for row in transitions))
        self.assertGreaterEqual(result["max_lift_m"], 0.0)
        self.assertFalse(result["success"])
        self.assertGreaterEqual(env.get_package_position()[2], env.config.table_surface_z - 0.02)

    def test_recorded_path_physically_lifts_and_places_package_in_tray(self):
        pickup = np.asarray((0.562, 0.100))
        dropoff = np.asarray((0.564, -0.197))
        env = ContactManipulationEnv(
            pickup_xyz=(pickup[0], pickup[1], 0.412),
            dropoff_xyz=(dropoff[0], dropoff[1], 0.416),
        )
        start = np.asarray((0.540, -0.050))
        xy_path = np.vstack((np.linspace(start, pickup, 6), np.linspace(pickup, dropoff, 30)[1:]))
        trajectory = np.column_stack((np.arange(len(xy_path)) * 0.1, xy_path, np.full(len(xy_path), 0.62)))
        result = run_demonstration_manipulation(env, trajectory, pickup, dropoff)
        self.assertTrue(result["success"], msg=result["failure_reason"])
        self.assertGreaterEqual(result["max_lift_m"], 0.05)
        self.assertTrue(result["stable_in_tray"])
        self.assertTrue(env.is_inside_tray())

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
