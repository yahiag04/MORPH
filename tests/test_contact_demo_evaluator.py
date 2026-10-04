"""Tests for repeatable contact-task evaluation and outcome aggregation."""

import unittest
import tempfile
from copy import copy
from pathlib import Path

import mujoco
import numpy as np

from evaluation.contact_demo_evaluator import (
    STATE_DIM, aggregate_contact_runs, contact_task_state, evaluate_contact_trajectory,
    sample_contact_control_rate,
    write_transition_archive,
)
from perception.task_layout import TaskLayout
from simulation.contact_manipulation_env import ContactManipulationEnv


class ContactDemoEvaluatorTests(unittest.TestCase):
    def test_contact_control_rate_keeps_timed_endpoints(self):
        trajectory = np.asarray([
            [0.0, .5, .0, .62], [.04, .51, .0, .62],
            [.10, .52, .0, .62], [.15, .53, .0, .62],
        ])

        sampled = sample_contact_control_rate(trajectory, 10.0)

        np.testing.assert_allclose(sampled[:, 0], (0.0, 0.10, 0.15))
        np.testing.assert_array_equal(sampled[-1], trajectory[-1])

    def test_observation_refreshes_derived_fields_without_changing_live_simulation(self):
        env = ContactManipulationEnv(pickup_xyz=(.55, .1, .412), dropoff_xyz=(.55, -.15, .416))
        env.data.ctrl[0] += .2
        for _ in range(10):
            env.step()
        saved = {key: getattr(env.data, key).copy() for key in
                 ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'ctrl', 'xpos')}
        time = env.data.time
        expected = copy(env)
        expected.data = mujoco.MjData(env.model)
        mujoco.mj_copyData(expected.data, env.model, env.data)
        mujoco.mj_forward(env.model, expected.data)
        self.assertGreater(np.max(np.abs(env.get_end_effector_position() -
                                        expected.get_end_effector_position())), 1e-7)
        state = contact_task_state(env)
        np.testing.assert_allclose(state[:3], expected.get_end_effector_position(), rtol=0, atol=3e-8)
        np.testing.assert_allclose(state[17:20], expected.get_package_position(), rtol=0, atol=3e-8)
        np.testing.assert_array_equal(state[32:34],
            [expected.package_has_gripper_contact(), expected.package_has_support_contact()])
        self.assertEqual(env.data.time, time)
        for key, value in saved.items():
            np.testing.assert_array_equal(getattr(env.data, key), value)

    def test_archive_keeps_timing_actuator_controls_and_complete_settling(self):
        pickup, dropoff = (.562, .1), (.564, -.197)
        path = np.array([[0., *pickup, .62], [1., .56, 0., .62], [2., *dropoff, .62]])
        run = evaluate_contact_trajectory(path, pickup, dropoff, human_label='synthetic',
            simulation_steps_per_sample=50, observation_steps=10)
        run.update(clip_id='episode-1', group_id='scenario-1', split='train', source_kind='simulation_randomized')
        rows = run['transitions']
        self.assertEqual(rows[-1]['phase'], 'settle')
        self.assertGreater(len(rows), run['control_steps'])
        self.assertAlmostEqual(sum(row['dt_seconds'] for row in rows), rows[-1]['simulation_end_time'])
        for left, right in zip(rows, rows[1:]):
            np.testing.assert_array_equal(left['next_state'], right['state'])
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'data.npz'
            write_transition_archive([run], out)
            with np.load(out, allow_pickle=False) as data:
                self.assertEqual(data['actuator_controls'].shape, (len(rows), 8))
                np.testing.assert_allclose(data['transition_dt_seconds'], .02)
                np.testing.assert_allclose(data['simulation_end_times']-data['simulation_start_times'], .02)
                self.assertEqual(set(data['source_kinds']), {'simulation_randomized'})
                self.assertEqual(set(data['group_ids']), {'scenario-1'})
                self.assertEqual(set(data['splits']), {'train'})
                self.assertEqual(set(data['episode_ids']), {'episode-1'})
                self.assertEqual(data['states'].shape[1], 37)

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
        observations = []
        first = evaluate_contact_trajectory(
            trajectory, pickup, dropoff, human_label="riusciti",
            on_observation=lambda env, row: observations.append((float(env.data.time), row)),
        )
        second = evaluate_contact_trajectory(trajectory, pickup, dropoff, human_label="riusciti")
        self.assertEqual(first["robot_success"], second["robot_success"])
        self.assertEqual(first["failure_reason"], second["failure_reason"])
        np.testing.assert_allclose(first["transitions"][0]["state"], second["transitions"][0]["state"])
        self.assertEqual(first["human_label"], "riusciti")
        self.assertEqual(len(observations), len(first["transitions"]) + 1)
        self.assertIsNone(observations[0][1])
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
