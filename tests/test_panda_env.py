"""Tests for the reusable Panda MuJoCo environment."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from simulation.panda_env import PandaEnv


ACTUATED_SCENE = """<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="arm">
      <joint name="shoulder" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body name="hand" pos="1 0 0">
        <geom type="sphere" size="0.1" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="shoulder" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""

NO_HAND_SCENE = """<mujoco>
  <worldbody><geom type="sphere" size="0.1"/></worldbody>
</mujoco>
"""


class PandaEnvTests(unittest.TestCase):
    def make_scene(self, directory: str, xml: str) -> Path:
        path = Path(directory) / "scene.xml"
        path.write_text(xml, encoding="utf-8")
        return path

    def test_reads_finite_three_dimensional_position_on_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            env = PandaEnv(self.make_scene(directory, ACTUATED_SCENE))

        position = env.get_end_effector_position()

        self.assertEqual(position.shape, (3,))
        self.assertTrue(np.isfinite(position).all())
        np.testing.assert_allclose(position, [1.0, 0.0, 0.0])

    def test_position_result_is_a_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            env = PandaEnv(self.make_scene(directory, ACTUATED_SCENE))

        position = env.get_end_effector_position()
        position[:] = 99.0

        np.testing.assert_allclose(env.get_end_effector_position(), [1.0, 0.0, 0.0])

    def test_missing_model_path_raises_file_not_found(self):
        missing_path = Path(tempfile.gettempdir()) / "missing-panda-scene.xml"

        with self.assertRaisesRegex(FileNotFoundError, str(missing_path)):
            PandaEnv(missing_path)

    def test_model_without_hand_body_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_scene(directory, NO_HAND_SCENE)

            with self.assertRaisesRegex(ValueError, "hand"):
                PandaEnv(path)

    def test_position_updates_after_control_and_simulation_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            env = PandaEnv(self.make_scene(directory, ACTUATED_SCENE))

        initial_position = env.get_end_effector_position()
        env.data.ctrl[0] = 1.0
        for _ in range(100):
            env.step()

        self.assertFalse(
            np.allclose(env.get_end_effector_position(), initial_position)
        )


if __name__ == "__main__":
    unittest.main()
