"""Tests for Cartesian waypoint interpolation, smoothing, and execution."""

import csv
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from simulation.trajectory_player import (
    CartesianTrajectoryPlayer,
    TrajectoryLog,
    resample_waypoints,
    smooth_trajectory,
)
from simulation.panda_env import PandaEnv


SAFE_SEED = np.array([0.0, -0.5, 0.0, -1.7, 0.0, 2.0, 0.0])
CURVE_CONFIGURATIONS = (
    SAFE_SEED,
    np.array([0.12, -0.42, 0.12, -1.82, 0.08, 2.1, 0.1]),
    np.array([0.22, -0.35, 0.2, -1.9, 0.15, 2.15, 0.2]),
)


def make_fixture_env(joint: str) -> PandaEnv:
    """Load a minimal valid scene for actuator mapping checks."""
    xml = f"""<mujoco>
      <worldbody>
        <body name="arm">
          {joint}
          <geom type="sphere" size="0.1" mass="1"/>
          <body name="hand" pos="1 0 0"/>
        </body>
      </worldbody>
    </mujoco>"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scene.xml"
        path.write_text(xml, encoding="utf-8")
        return PandaEnv(path)


class ResampleWaypointsTests(unittest.TestCase):
    def test_samples_by_cumulative_distance_at_configured_speed(self):
        waypoints = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0]]
        )

        times, samples = resample_waypoints(
            waypoints, speed=0.1, sample_period=0.5
        )

        np.testing.assert_allclose(times, [0.0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(
            samples,
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.1, 0.05, 0.0],
                [0.1, 0.1, 0.0],
            ],
        )

    def test_ignores_repeated_interior_waypoint(self):
        waypoints = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0]]
        )

        times, samples = resample_waypoints(
            waypoints, speed=0.1, sample_period=0.5
        )

        np.testing.assert_allclose(times, [0.0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(samples[2], [0.1, 0.0, 0.0])

    def test_rejects_waypoints_with_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            resample_waypoints(np.array([[0.0, 1.0], [1.0, 2.0]]), 0.1, 0.02)

    def test_rejects_non_finite_waypoints(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            resample_waypoints(
                np.array([[0.0, 0.0, 0.0], [np.inf, 0.0, 0.0]]), 0.1, 0.02
            )

    def test_rejects_fewer_than_two_waypoints(self):
        with self.assertRaisesRegex(ValueError, "two"):
            resample_waypoints(np.array([[0.0, 0.0, 0.0]]), 0.1, 0.02)

    def test_rejects_zero_length_path(self):
        with self.assertRaisesRegex(ValueError, "distance"):
            resample_waypoints(
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), 0.1, 0.02
            )

    def test_preserves_both_endpoints_for_extremely_short_path(self):
        waypoints = np.array([[0.0, 0.0, 0.0], [1e-14, 0.0, 0.0]])

        times, samples = resample_waypoints(waypoints, speed=1.0, sample_period=0.02)

        np.testing.assert_array_equal(times, [0.0, 1e-14])
        np.testing.assert_array_equal(samples, waypoints)

    def test_collapses_numerically_duplicate_final_sample(self):
        waypoints = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1 + 0.2, 0.0, 0.0]])

        times, _ = resample_waypoints(waypoints, speed=1.0, sample_period=0.02)

        self.assertEqual(times[-1], 0.1 + 0.2)
        self.assertGreater(np.diff(times).min(), 1e-12)

    def test_rejects_nonpositive_speed_or_sample_period(self):
        path = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        for speed, period in ((0.0, 0.02), (0.1, 0.0), (0.1, -0.02)):
            with self.subTest(speed=speed, period=period):
                with self.assertRaisesRegex(ValueError, "speed|sample_period"):
                    resample_waypoints(path, speed, period)


class SmoothTrajectoryTests(unittest.TestCase):
    def test_smooths_interior_and_preserves_endpoints(self):
        samples = np.array(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0],
             [3.0, 2.0, 0.0], [4.0, 0.0, 0.0]]
        )

        smoothed = smooth_trajectory(samples, window_size=3)

        np.testing.assert_array_equal(smoothed[0], samples[0])
        np.testing.assert_array_equal(smoothed[-1], samples[-1])
        self.assertAlmostEqual(smoothed[1, 1], 2.0 / 3.0)
        self.assertEqual(smoothed.shape, samples.shape)

    def test_caps_oversized_window_to_available_odd_length(self):
        samples = np.array(
            [[0.0, 0.0, 0.0], [1.0, 3.0, 0.0], [2.0, 0.0, 0.0]]
        )

        smoothed = smooth_trajectory(samples, window_size=9)

        np.testing.assert_array_equal(smoothed[0], samples[0])
        np.testing.assert_array_equal(smoothed[-1], samples[-1])
        self.assertEqual(smoothed.shape, samples.shape)
        self.assertAlmostEqual(smoothed[1, 1], 1.0)

    def test_window_one_disables_smoothing(self):
        samples = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
        np.testing.assert_array_equal(smooth_trajectory(samples, 1), samples)

    def test_rejects_invalid_window_size(self):
        samples = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        for window in (0, 2, -1):
            with self.subTest(window=window):
                with self.assertRaisesRegex(ValueError, "window_size"):
                    smooth_trajectory(samples, window)


class CartesianTrajectoryPlayerTests(unittest.TestCase):
    def test_validates_speed_and_control_period_alignment(self):
        env = PandaEnv()
        for kwargs, message in (
            ({"speed": 0.0}, "speed"),
            ({"control_period": 0.001}, "control_period"),
            ({"control_period": 0.003}, "control_period"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    CartesianTrajectoryPlayer(env, **kwargs)

    def test_rejects_missing_position_actuator_by_name(self):
        env = make_fixture_env(
            '<joint name="joint1" type="hinge" limited="true" range="-1 1"/>'
        )
        with self.assertRaisesRegex(ValueError, "actuator1"):
            CartesianTrajectoryPlayer(env)

    def test_rejects_position_actuator_with_nonunit_gear(self):
        env = PandaEnv()
        env.model.actuator_gear[0, 0] = 2.0

        with self.assertRaisesRegex(ValueError, "gear"):
            CartesianTrajectoryPlayer(env)

    def test_follows_cartesian_curve_and_writes_physics_log(self):
        env = PandaEnv()
        waypoints = []
        for q in CURVE_CONFIGURATIONS:
            env.data.qpos[:7] = q
            mujoco.mj_forward(env.model, env.data)
            waypoints.append(env.get_end_effector_position())

        start_q = CURVE_CONFIGURATIONS[0]
        env.data.qpos[:7] = start_q
        env.data.ctrl[:7] = start_q
        mujoco.mj_forward(env.model, env.data)
        env.data.ctrl[7] = 0.0

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nested" / "trajectory.csv"
            log = CartesianTrajectoryPlayer(
                env, speed=0.08, control_period=0.02, smoothing_window=1
            ).follow(np.array(waypoints), output_csv=output_path)

            self.assertTrue(output_path.is_file())
            with output_path.open(newline="", encoding="utf-8") as stream:
                header = next(csv.reader(stream))

        self.assertIsInstance(log, TrajectoryLog)
        self.assertGreater(log.timestamps.size, 0)
        self.assertEqual(log.target_xyz.shape, log.actual_xyz.shape)
        self.assertEqual(log.target_xyz.shape[1], 3)
        self.assertTrue(np.isfinite(log.timestamps).all())
        self.assertTrue(np.isfinite(log.target_xyz).all())
        self.assertTrue(np.isfinite(log.actual_xyz).all())
        self.assertTrue(np.all(np.diff(log.timestamps) > 0.0))
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(
            log.actual_xyz[-1], env.get_end_effector_position(), atol=1e-12
        )
        self.assertLess(
            np.linalg.norm(log.target_xyz[-1] - log.actual_xyz[-1]), 0.05
        )
        self.assertTrue(log.ik_converged.all())
        self.assertEqual(env.data.ctrl[7], 0.0)
        self.assertTrue(
            np.all(env.data.qpos[7:9] >= env.model.jnt_range[7:9, 0])
        )
        self.assertTrue(
            np.all(env.data.qpos[7:9] <= env.model.jnt_range[7:9, 1])
        )
        self.assertFalse(np.allclose(env.data.qpos[:7], start_q))
        self.assertEqual(
            header,
            [
                "timestamp",
                "target_x",
                "target_y",
                "target_z",
                "actual_x",
                "actual_y",
                "actual_z",
                "ik_converged",
            ],
        )

    def test_logs_ik_nonconvergence_and_continues(self):
        env = PandaEnv()
        start = env.get_end_effector_position()
        waypoints = np.vstack((start, [0.09, 0.0, 1.6]))

        log = CartesianTrajectoryPlayer(
            env, speed=100.0, control_period=env.model.opt.timestep, smoothing_window=1
        ).follow(waypoints)

        self.assertGreater(log.timestamps.size, 0)
        self.assertFalse(log.ik_converged.all())
        self.assertTrue(np.isfinite(log.actual_xyz).all())


if __name__ == "__main__":
    unittest.main()
