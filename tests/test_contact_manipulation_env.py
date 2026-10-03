"""Tests for the package-and-tray contact task."""

import unittest

import mujoco
import numpy as np

from simulation.contact_manipulation_env import ContactManipulationEnv
from simulation.panda_env import DEFAULT_MODEL_PATH


class ContactManipulationEnvTests(unittest.TestCase):
    def make_env(self):
        surface_z = 0.4
        return ContactManipulationEnv(
            pickup_xyz=(0.52, -0.12, surface_z + 0.012),
            dropoff_xyz=(0.58, 0.12, surface_z + 0.016),
        )

    def test_builds_free_package_and_collision_enabled_tray(self):
        env = self.make_env()
        for kind, name in (
            (mujoco.mjtObj.mjOBJ_BODY, "package"),
            (mujoco.mjtObj.mjOBJ_JOINT, "package_free"),
            (mujoco.mjtObj.mjOBJ_GEOM, "package_geom"),
            (mujoco.mjtObj.mjOBJ_GEOM, "tray_floor"),
            (mujoco.mjtObj.mjOBJ_GEOM, "tray_wall_front"),
            (mujoco.mjtObj.mjOBJ_GEOM, "tray_wall_back"),
            (mujoco.mjtObj.mjOBJ_GEOM, "tray_wall_left"),
            (mujoco.mjtObj.mjOBJ_GEOM, "tray_wall_right"),
        ):
            with self.subTest(name=name):
                self.assertGreaterEqual(mujoco.mj_name2id(env.model, kind, name), 0)

        package_geom = env.package_geom_id
        np.testing.assert_allclose(env.model.geom_size[package_geom], (0.025, 0.020, 0.012))
        self.assertAlmostEqual(env.model.body_mass[env.package_body_id], 0.02)
        np.testing.assert_allclose(env.data.qpos[:7], (0.0, -0.5, 0.0, -1.7, 0.0, 2.0, 0.0))
        for name in ("tray_floor", "tray_wall_front", "tray_wall_back", "tray_wall_left", "tray_wall_right"):
            geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.assertGreater(env.model.geom_contype[geom_id], 0)
            self.assertGreater(env.model.geom_conaffinity[geom_id], 0)
        self.assertEqual(env.model.jnt_type[env.package_joint_id], mujoco.mjtJoint.mjJNT_FREE)

    def test_initializes_package_at_pickup_and_names_gripper_actuator(self):
        env = self.make_env()
        np.testing.assert_allclose(env.get_package_position(), (0.52, -0.12, 0.412), atol=1e-12)
        self.assertEqual(
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, env.gripper_actuator_id),
            "actuator8",
        )
        env.open_gripper()
        self.assertEqual(
            env.data.ctrl[env.gripper_actuator_id],
            env.model.actuator_ctrlrange[env.gripper_actuator_id, 1],
        )
        env.close_gripper()
        self.assertEqual(
            env.data.ctrl[env.gripper_actuator_id],
            env.model.actuator_ctrlrange[env.gripper_actuator_id, 0],
        )

    def test_rejects_nonfinite_and_out_of_workspace_pickup_or_dropoff(self):
        valid_pickup = (0.52, -0.12, 0.412)
        valid_dropoff = (0.58, 0.12, 0.416)
        with self.assertRaisesRegex(ValueError, "finite"):
            ContactManipulationEnv(pickup_xyz=(np.nan, 0.0, 0.412), dropoff_xyz=valid_dropoff)
        with self.assertRaisesRegex(ValueError, "workspace"):
            ContactManipulationEnv(pickup_xyz=(1.2, 0.0, 0.412), dropoff_xyz=valid_dropoff)
        with self.assertRaisesRegex(ValueError, "workspace"):
            ContactManipulationEnv(pickup_xyz=valid_pickup, dropoff_xyz=(0.58, 0.7, 0.406))

    def test_reports_package_finger_and_tray_support_contacts(self):
        env = self.make_env()
        for _ in range(20):
            env.step()
        self.assertFalse(env.package_has_gripper_contact())
        self.assertTrue(env.package_has_support_contact())
        self.assertTrue(env.is_package_stable_on_support())


if __name__ == "__main__":
    unittest.main()
