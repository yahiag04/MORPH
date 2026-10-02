"""Tests for damped least-squares Panda position IK."""

from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from simulation.ik import IKResult, solve_position_ik
from simulation.panda_env import PandaEnv


SAFE_SEED = np.array([0.0, -0.5, 0.0, -1.7, 0.0, 2.0, 0.0])
GOAL_CONFIGURATIONS = (
    np.array([0.15, -0.4, 0.2, -1.9, 0.1, 2.1, -0.1]),
    np.array([-0.25, -0.35, 0.35, -2.0, -0.2, 2.3, 0.25]),
    np.array([0.3, -0.55, -0.25, -1.6, 0.15, 2.1, -0.3]),
)


def make_fixture_env(joint: str) -> PandaEnv:
    """Load a tiny valid scene with the supplied arm joint declaration."""
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


class IKValidationTests(unittest.TestCase):
    def setUp(self):
        self.env = PandaEnv()

    def test_rejects_target_with_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "three"):
            solve_position_ik(self.env, np.array([0.1, 0.2]))

    def test_rejects_non_finite_target(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            solve_position_ik(self.env, np.array([0.1, np.nan, 0.3]))

    def test_rejects_nonpositive_tolerance_and_damping(self):
        target = np.array([0.2, 0.1, 0.5])
        for parameter, value in (("tolerance", 0.0), ("damping", -0.01)):
            with self.subTest(parameter=parameter):
                with self.assertRaisesRegex(ValueError, parameter):
                    solve_position_ik(self.env, target, **{parameter: value})

    def test_rejects_non_numeric_solver_parameters(self):
        target = np.array([0.2, 0.1, 0.5])
        for parameter, value in (
            ("tolerance", "slow"),
            ("damping", None),
            ("step_size", "half"),
        ):
            with self.subTest(parameter=parameter):
                with self.assertRaisesRegex(ValueError, parameter):
                    solve_position_ik(self.env, target, **{parameter: value})

    def test_rejects_nonpositive_iteration_limit(self):
        with self.assertRaisesRegex(ValueError, "max_iterations"):
            solve_position_ik(
                self.env, np.array([0.2, 0.1, 0.5]), max_iterations=0
            )

    def test_rejects_step_size_outside_unit_interval(self):
        for step_size in (0.0, 1.01):
            with self.subTest(step_size=step_size):
                with self.assertRaisesRegex(ValueError, "step_size"):
                    solve_position_ik(
                        self.env,
                        np.array([0.2, 0.1, 0.5]),
                        step_size=step_size,
                    )

    def test_rejects_missing_arm_joint_by_name(self):
        env = make_fixture_env(
            '<joint name="other_joint" type="hinge" limited="true" range="-1 1"/>'
        )
        with self.assertRaisesRegex(ValueError, "joint1"):
            solve_position_ik(env, np.array([0.5, 0.0, 0.0]))

    def test_rejects_non_hinge_arm_joint(self):
        env = make_fixture_env(
            '<joint name="joint1" type="slide" limited="true" range="-1 1"/>'
        )
        with self.assertRaisesRegex(ValueError, "joint1"):
            solve_position_ik(env, np.array([0.5, 0.0, 0.0]))

    def test_rejects_unlimited_arm_joint(self):
        env = make_fixture_env(
            '<joint name="joint1" type="hinge" limited="false"/>'
        )
        with self.assertRaisesRegex(ValueError, "joint1"):
            solve_position_ik(env, np.array([0.5, 0.0, 0.0]))


class IKConvergenceTests(unittest.TestCase):
    def setUp(self):
        self.env = PandaEnv()

    def set_arm_configuration(self, q: np.ndarray) -> None:
        self.env.data.qpos[:7] = q
        mujoco.mj_forward(self.env.model, self.env.data)

    def test_converges_to_multiple_reachable_xyz_targets(self):
        finger_qpos = np.array([0.025, 0.035])
        self.env.data.qpos[7:9] = finger_qpos

        for goal_q in GOAL_CONFIGURATIONS:
            with self.subTest(goal=goal_q.tolist()):
                self.set_arm_configuration(goal_q)
                target = self.env.get_end_effector_position()
                self.set_arm_configuration(SAFE_SEED)

                result = solve_position_ik(self.env, target)

                self.assertIsInstance(result, IKResult)
                self.assertTrue(result.converged, result)
                self.assertLessEqual(result.position_error, 0.015)
                self.assertEqual(result.q.shape, (7,))
                self.assertTrue(np.isfinite(result.q).all())
                self.assertGreater(result.iterations, 0)
                np.testing.assert_allclose(
                    self.env.get_end_effector_position(),
                    target,
                    atol=result.position_error + 1e-10,
                )
                np.testing.assert_allclose(self.env.data.qpos[7:9], finger_qpos)
                for joint_index in range(7):
                    lower, upper = self.env.model.jnt_range[joint_index]
                    self.assertGreaterEqual(result.q[joint_index], lower)
                    self.assertLessEqual(result.q[joint_index], upper)

    def test_clamps_initial_qpos_before_solving(self):
        self.env.data.qpos[3] = 0.0  # Panda joint4 upper limit is negative.
        target = np.array([0.5, 0.1, 0.7])

        result = solve_position_ik(self.env, target, max_iterations=1)

        self.assertLessEqual(result.q[3], self.env.model.jnt_range[3, 1])
        self.assertTrue(np.isfinite(result.position_error))

    def test_copies_target_view_before_forward_kinematics_updates_data(self):
        target_view = self.env.data.xpos[self.env.hand_id]
        original_target = target_view.copy()

        result = solve_position_ik(self.env, target_view)

        actual_error = np.linalg.norm(
            original_target - self.env.get_end_effector_position()
        )
        self.assertTrue(result.converged)
        self.assertLessEqual(actual_error, 0.015)
        self.assertAlmostEqual(result.position_error, actual_error, places=9)

    def test_reports_nonconvergence_for_far_target(self):
        self.set_arm_configuration(SAFE_SEED)

        result = solve_position_ik(
            self.env,
            np.array([10.0, 10.0, 10.0]),
            max_iterations=1,
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertTrue(np.isfinite(result.position_error))
        self.assertGreater(result.position_error, 0.015)


if __name__ == "__main__":
    unittest.main()
