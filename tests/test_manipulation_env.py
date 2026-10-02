"""Tests for the physical Panda manipulation scene."""

import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from simulation.manipulation_env import ManipulationEnv
from simulation.panda_env import DEFAULT_MODEL_PATH


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

    def test_missing_gripper_actuator_has_clear_error(self):
        spec = mujoco.MjSpec.from_file(str(DEFAULT_MODEL_PATH))
        spec.delete(spec.actuator("actuator8"))
        key = spec.key("home")
        key.ctrl = [key.ctrl[index] for index in range(7)]

        with patch(
            "simulation.manipulation_env.mujoco.MjSpec.from_file",
            return_value=spec,
        ):
            with self.assertRaisesRegex(ValueError, "actuator8"):
                ManipulationEnv()

    def test_cube_requires_table_contact_to_be_supported(self):
        env = ManipulationEnv()
        cube_joint = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free"
        )
        cube_qpos = env.model.jnt_qposadr[cube_joint]
        env.data.qpos[cube_qpos + 2] += 0.02
        mujoco.mj_forward(env.model, env.data)

        self.assertFalse(env.cube_has_table_contact())
        self.assertFalse(env.is_cube_stably_on_table())

    def test_cube_must_be_nearly_stationary_to_be_placed(self):
        env = ManipulationEnv()
        cube_joint = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free"
        )
        cube_dof = env.model.jnt_dofadr[cube_joint]
        cube_qpos = env.model.jnt_qposadr[cube_joint]
        env.data.qpos[cube_qpos + 2] = (
            env.config.table_surface_z + env.config.cube_half_size - 0.0001
        )
        env.data.qvel[cube_dof] = 0.1
        mujoco.mj_forward(env.model, env.data)

        self.assertTrue(env.cube_has_table_contact())
        self.assertFalse(env.is_cube_stably_on_table())


if __name__ == "__main__":
    unittest.main()
