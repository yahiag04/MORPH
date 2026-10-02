"""Minimal Panda scene for physical cube pick-and-place experiments."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .panda_env import DEFAULT_MODEL_PATH, PandaEnv


@dataclass(frozen=True)
class ManipulationConfig:
    """World dimensions and initial positions for the manipulation scene."""

    cube_start_xyz: tuple[float, float, float] = (0.55, 0.15, 0.425)
    target_xyz: tuple[float, float, float] = (0.55, -0.15, 0.425)
    table_center_xyz: tuple[float, float, float] = (0.55, 0.0, 0.35)
    table_half_size_xyz: tuple[float, float, float] = (0.35, 0.4, 0.05)
    cube_half_size: float = 0.025
    target_half_size_xy: float = 0.06
    cube_yaw_radians: float = 1.0471975511965976
    gripper_center_offset_xy: tuple[float, float] = (0.008, 0.003)

    @property
    def table_surface_z(self) -> float:
        """Height of the tabletop in world coordinates."""
        return self.table_center_xyz[2] + self.table_half_size_xyz[2]


class ManipulationEnv(PandaEnv):
    """Panda environment containing a table, free cube, and target patch."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        config: ManipulationConfig | None = None,
    ) -> None:
        self.config = config or ManipulationConfig()
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MuJoCo scene file not found: {path}")
        spec = mujoco.MjSpec.from_file(str(path))
        world = spec.worldbody

        world.add_geom(
            name="table_surface",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=self.config.table_center_xyz,
            size=self.config.table_half_size_xyz,
            rgba=(0.32, 0.22, 0.14, 1.0),
            friction=(1.0, 0.01, 0.001),
        )
        target_xyz = self.config.target_xyz
        world.add_geom(
            name="target_region",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(target_xyz[0], target_xyz[1], self.config.table_surface_z + 0.001),
            size=(self.config.target_half_size_xy, self.config.target_half_size_xy, 0.001),
            rgba=(0.15, 0.8, 0.25, 0.65),
            contype=0,
            conaffinity=0,
        )
        cube = world.add_body(
            name="cube",
            pos=self.config.cube_start_xyz,
            quat=(
                np.cos(self.config.cube_yaw_radians / 2.0),
                0.0,
                0.0,
                np.sin(self.config.cube_yaw_radians / 2.0),
            ),
        )
        cube.add_freejoint(name="cube_free")
        cube.add_geom(
            name="cube_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.config.cube_half_size,) * 3,
            mass=0.08,
            rgba=(0.9, 0.32, 0.08, 1.0),
            friction=(1.2, 0.01, 0.001),
            condim=4,
        )

        super().__init__(model=spec.compile())
        self.cube_body_id = self._required_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )
        self.cube_geom_id = self._required_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"
        )
        self.target_geom_id = self._required_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_region"
        )
        self.table_geom_id = self._required_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_surface"
        )
        self.gripper_actuator_id = self._required_id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
        )

    @staticmethod
    def _required_id(
        model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str
    ) -> int:
        """Resolve a required named element with a model-specific message."""
        identifier = mujoco.mj_name2id(model, object_type, name)
        if identifier < 0:
            role = "gripper " if name == "actuator8" else ""
            raise ValueError(f"Manipulation model is missing {role}element '{name}'")
        return identifier

    def get_cube_position(self) -> np.ndarray:
        """Return a copy of the cube center in world coordinates."""
        return self.data.xpos[self.cube_body_id].copy()

    def is_in_target_region(self, xyz: np.ndarray) -> bool:
        """Check whether world-frame XYZ falls inside the target's XY bounds."""
        point = np.asarray(xyz, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("xyz must contain three finite coordinates")
        half_width = self.config.target_half_size_xy
        return bool(np.all(np.abs(point[:2] - self.config.target_xyz[:2]) <= half_width))

    def is_cube_stably_on_table(
        self,
        *,
        max_linear_speed: float = 0.02,
        max_angular_speed: float = 0.5,
    ) -> bool:
        """Require real table contact and low translational/angular velocity."""
        if not self.cube_has_table_contact():
            return False
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.cube_body_id,
            velocity,
            0,
        )
        return bool(
            np.linalg.norm(velocity[3:]) <= max_linear_speed
            and np.linalg.norm(velocity[:3]) <= max_angular_speed
        )

    def cube_has_table_contact(self) -> bool:
        """Return whether a current MuJoCo contact supports the cube on the table."""
        return any(
            {int(contact.geom1), int(contact.geom2)}
            == {self.cube_geom_id, self.table_geom_id}
            and contact.dist <= 0.001
            for contact in self.data.contact[: self.data.ncon]
        )

    def open_gripper(self) -> None:
        """Command the Panda gripper to its maximum actuator setting."""
        self.data.ctrl[self.gripper_actuator_id] = self.model.actuator_ctrlrange[
            self.gripper_actuator_id, 1
        ]

    def close_gripper(self) -> None:
        """Command the Panda gripper to its minimum actuator setting."""
        self.data.ctrl[self.gripper_actuator_id] = self.model.actuator_ctrlrange[
            self.gripper_actuator_id, 0
        ]
