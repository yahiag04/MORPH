"""Physical integration tests for the scripted oracle task."""

import unittest

import mujoco
import numpy as np

from simulation.manipulation_env import ManipulationEnv
from simulation.pick_place import run_oracle_pick_and_place


class OraclePickPlaceTests(unittest.TestCase):
    def test_oracle_physically_lifts_and_places_cube_in_target(self):
        env = ManipulationEnv()
        seed = np.array([0.0, -0.5, 0.0, -1.7, 0.0, 2.0, 0.0])
        env.data.qpos[:7] = seed
        env.data.ctrl[:7] = seed
        env.open_gripper()
        mujoco.mj_forward(env.model, env.data)
        start_xyz = env.get_cube_position()
        callback_times = []

        result = run_oracle_pick_and_place(
            env, on_control_step=lambda: callback_times.append(env.data.time)
        )

        self.assertTrue(result.object_lifted, msg=f"peak={result.max_cube_z}")
        self.assertTrue(env.is_in_target_region(result.final_cube_xyz))
        self.assertTrue(result.success)
        self.assertTrue(env.is_cube_stably_on_table())
        self.assertGreater(result.log.timestamps.size, 0)
        self.assertTrue(np.all(np.diff(result.log.timestamps) > 0.0))
        np.testing.assert_allclose(
            callback_times[:15], np.arange(0.02, 0.301, 0.02), atol=1e-12
        )
        self.assertLess(np.linalg.norm(result.initial_cube_xyz - start_xyz), 1e-12)


if __name__ == "__main__":
    unittest.main()
