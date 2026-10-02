"""Tests for the physical Panda manipulation scene."""

import unittest

import mujoco
import numpy as np

from simulation.manipulation_env import ManipulationEnv


class ManipulationEnvTests(unittest.TestCase):
    def test_builds_table_cube_target_and_places_cube_on_table(self):
        env = ManipulationEnv()

        for kind, name in (
            (mujoco.mjtObj.mjOBJ_GEOM, "table_surface"),
            (mujoco.mjtObj.mjOBJ_BODY, "cube"),
            (mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"),
            (mujoco.mjtObj.mjOBJ_GEOM, "target_region"),
        ):
            with self.subTest(name=name):
                self.assertGreaterEqual(mujoco.mj_name2id(env.model, kind, name), 0)

        cube_body = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )
        cube_joint = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free"
        )
        self.assertEqual(env.model.jnt_type[cube_joint], mujoco.mjtJoint.mjJNT_FREE)
        np.testing.assert_allclose(
            env.data.xpos[cube_body], env.config.cube_start_xyz, atol=1e-12
        )
        self.assertTrue(env.is_in_target_region(env.config.target_xyz))
        self.assertFalse(env.is_in_target_region(env.config.cube_start_xyz))

    def test_gripper_commands_use_named_actuator_limits(self):
        env = ManipulationEnv()
        env.data.ctrl[:7] = np.arange(7, dtype=float)
        arm_control = env.data.ctrl[:7].copy()

        env.open_gripper()
        actuator_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
        )
        self.assertEqual(env.data.ctrl[actuator_id], env.model.actuator_ctrlrange[actuator_id, 1])
        np.testing.assert_array_equal(env.data.ctrl[:7], arm_control)

        env.close_gripper()
        self.assertEqual(env.data.ctrl[actuator_id], env.model.actuator_ctrlrange[actuator_id, 0])
        np.testing.assert_array_equal(env.data.ctrl[:7], arm_control)


if __name__ == "__main__":
    unittest.main()
