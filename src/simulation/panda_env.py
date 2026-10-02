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

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        model: mujoco.MjModel | None = None,
    ) -> None:
        """Load ``model_path`` and resolve the body used as the end effector.

        Args:
            model_path: MuJoCo XML scene path. Defaults to the local Menagerie
                Panda scene.
            model: Optional precompiled MuJoCo model. If supplied, it is used
                directly and ``model_path`` is ignored.

        Raises:
            FileNotFoundError: If the scene path does not exist.
            ValueError: If the loaded model has no body named ``hand``.
        """
        scene_path = Path(model_path).expanduser()
        if model is None:
            if not scene_path.is_file():
                raise FileNotFoundError(f"MuJoCo scene file not found: {scene_path}")
            model = mujoco.MjModel.from_xml_path(str(scene_path))

        self.model = model
        self.data = mujoco.MjData(self.model)
        self.hand_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )
        if self.hand_id < 0:
            raise ValueError(
                f"MuJoCo model '{scene_path}' has no body named 'hand'"
            )

        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        """Advance the simulation by one MuJoCo timestep."""
        mujoco.mj_step(self.model, self.data)

    def get_end_effector_position(self) -> np.ndarray:
        """Return a copy of the hand body's world-frame XYZ position."""
        return self.data.xpos[self.hand_id].copy()
