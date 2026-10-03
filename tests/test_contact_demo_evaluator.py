"""Tests for repeatable contact-task evaluation and outcome aggregation."""

import unittest

import mujoco
import numpy as np

from evaluation.contact_demo_evaluator import (
    STATE_DIM, aggregate_contact_runs, contact_task_state, evaluate_contact_trajectory,
)
from perception.task_layout import TaskLayout
from simulation.contact_manipulation_env import ContactManipulationEnv


class ContactDemoEvaluatorTests(unittest.TestCase):
    def test_complete_state_includes_robot_package_and_task_context(self):
        env = ContactManipulationEnv(
            pickup_xyz=(0.55, 0.0, 0.412), dropoff_xyz=(0.55, 0.14, 0.416)
        )
        for joint_index in range(1, 8):
            joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{joint_index}")
            env.data.qvel[env.model.jnt_dofadr[joint_id]] = 0.01 * joint_index
        package_dof = env.model.jnt_dofadr[env.package_joint_id]
        env.data.qvel[package_dof:package_dof + 6] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

        state = contact_task_state(env)

        self.assertEqual(STATE_DIM, 37)
        self.assertEqual(state.shape, (37,))
        self.assertTrue(np.isfinite(state).all())
        np.testing.assert_allclose(state[10:17], np.arange(1, 8) * 0.01)
        np.testing.assert_allclose(state[20:24], env.data.qpos[env.model.jnt_qposadr[env.package_joint_id] + 3:
                                                                  env.model.jnt_qposadr[env.package_joint_id] + 7])
        np.testing.assert_allclose(state[24:27], (0.1, 0.2, 0.3))
        np.testing.assert_allclose(state[27:30], (0.4, 0.5, 0.6))
        np.testing.assert_allclose(state[34:37], env.dropoff_xyz)

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
