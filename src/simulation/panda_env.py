"""Small MuJoCo environment wrapper for reading Panda end-effector state."""

from pathlib import Path

import mujoco
import numpy as np


DEFAULT_MODEL_PATH = Path(
    "/Users/yahiaghallale/Documents/mujoco_menagerie/"
    "franka_emika_panda/scene.xml"
)


class PandaEnv:
    """Load a Panda scene and expose simulation state for simple scripts."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        """Load ``model_path`` and resolve the body used as the end effector.

        Args:
            model_path: MuJoCo XML scene path. Defaults to the local Menagerie
                Panda scene documented in ``AGENT_TASK.md``.

        Raises:
            FileNotFoundError: If the scene path does not exist.
            ValueError: If the loaded model has no body named ``hand``.
        """
        scene_path = Path(model_path).expanduser()
        if not scene_path.is_file():
            raise FileNotFoundError(f"MuJoCo scene file not found: {scene_path}")

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.hand_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )
        if self.hand_id < 0:
            raise ValueError(
                f"MuJoCo scene '{scene_path}' has no body named 'hand'"
            )

        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        """Advance the simulation by one MuJoCo timestep."""
        mujoco.mj_step(self.model, self.data)

    def get_end_effector_position(self) -> np.ndarray:
        """Return a copy of the hand body's world-frame XYZ position."""
        return self.data.xpos[self.hand_id].copy()
