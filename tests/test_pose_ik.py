"""Tests for position and orientation constrained Panda IK."""

import unittest

import mujoco
import numpy as np

from simulation.contact_manipulation_env import ContactManipulationEnv
from simulation.ik import solve_pose_ik, solve_position_ik


class PoseIKTests(unittest.TestCase):
    def test_converges_to_reachable_pose_and_preserves_orientation(self):
        env = ContactManipulationEnv(pickup_xyz=(0.55, 0.15, 0.412), dropoff_xyz=(0.55, -0.15, 0.416))
        target_xyz = np.array((0.55, 0.15, 0.66))
        qpos_before = env.data.qpos.copy()
        position_result = solve_position_ik(env, target_xyz)
        target_quat = env.data.xquat[env.hand_id].copy()
        env.data.qpos[:] = qpos_before
        mujoco.mj_forward(env.model, env.data)
        result = solve_pose_ik(env, target_xyz, target_quat)
        self.assertTrue(result.converged)
        self.assertLess(result.position_error, 0.005)
        quaternion_error = np.zeros(3)
        mujoco.mju_subQuat(quaternion_error, target_quat, env.data.xquat[env.hand_id])
        self.assertLess(np.linalg.norm(quaternion_error), 0.02)

    def test_rejects_invalid_quaternion(self):
        env = ContactManipulationEnv(pickup_xyz=(0.55, 0.15, 0.412), dropoff_xyz=(0.55, -0.15, 0.416))
        with self.assertRaisesRegex(ValueError, "quaternion"):
            solve_pose_ik(env, np.array((0.55, 0.15, 0.66)), np.array((0.0, 0.0, 0.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
