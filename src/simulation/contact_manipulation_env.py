"""MuJoCo scene for a small package and a collision-enabled receiving tray."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .panda_env import DEFAULT_MODEL_PATH, PandaEnv


@dataclass(frozen=True)
class ContactTaskConfig:
    """Dimensions and contact properties of the table, package, and tray."""

    table_center_xyz: tuple[float, float, float] = (0.55, 0.0, 0.35)
    table_half_size_xyz: tuple[float, float, float] = (0.35, 0.4, 0.05)
    package_half_size_xyz: tuple[float, float, float] = (0.025, 0.020, 0.012)
    package_mass: float = 0.06
    tray_inner_half_size_xy: float = 0.055
    tray_wall_thickness: float = 0.010
    tray_wall_height: float = 0.035
    tray_floor_thickness: float = 0.004
    gripper_center_offset_xy: tuple[float, float] = (0.008, 0.003)

    @property
    def table_surface_z(self) -> float:
        return self.table_center_xyz[2] + self.table_half_size_xyz[2]

    @property
    def tray_floor_top_z(self) -> float:
        return self.table_surface_z + self.tray_floor_thickness

    @property
    def package_resting_z(self) -> float:
        return self.tray_floor_top_z + self.package_half_size_xyz[2]


class ContactManipulationEnv(PandaEnv):
    """A freely moving package and square tray using ordinary MuJoCo contacts."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        pickup_xyz: tuple[float, float, float],
        dropoff_xyz: tuple[float, float, float],
        config: ContactTaskConfig | None = None,
    ) -> None:
        self.config = config or ContactTaskConfig()
        self.pickup_xyz = self._validate_task_point("pickup_xyz", pickup_xyz)
        self.dropoff_xyz = self._validate_task_point("dropoff_xyz", dropoff_xyz)
        self._validate_task_geometry()

        scene_path = Path(model_path).expanduser()
        if not scene_path.is_file():
            raise FileNotFoundError(f"MuJoCo scene file not found: {scene_path}")
        spec = mujoco.MjSpec.from_file(str(scene_path))
        world = spec.worldbody
        world.add_geom(
            name="contact_table_surface",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=self.config.table_center_xyz,
            size=self.config.table_half_size_xyz,
            rgba=(0.32, 0.22, 0.14, 1.0),
            friction=(1.0, 0.01, 0.001),
        )
        tray_x, tray_y = self.dropoff_xyz[:2]
        inner = self.config.tray_inner_half_size_xy
        wall = self.config.tray_wall_thickness
        floor_top = self.config.tray_floor_top_z
        floor_half_z = self.config.tray_floor_thickness / 2.0
        world.add_geom(
            name="tray_floor",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(tray_x, tray_y, self.config.table_surface_z + floor_half_z),
            size=(inner + wall, inner + wall, floor_half_z),
            rgba=(0.55, 0.36, 0.18, 1.0),
            friction=(1.2, 0.02, 0.002),
        )
        wall_z = floor_top + self.config.tray_wall_height / 2.0
        wall_height = self.config.tray_wall_height / 2.0
        walls = (
            ("front", (tray_x, tray_y - inner - wall, wall_z), (inner + wall, wall, wall_height)),
            ("back", (tray_x, tray_y + inner + wall, wall_z), (inner + wall, wall, wall_height)),
            ("left", (tray_x - inner - wall, tray_y, wall_z), (wall, inner, wall_height)),
            ("right", (tray_x + inner + wall, tray_y, wall_z), (wall, inner, wall_height)),
        )
        for side, position, size in walls:
            world.add_geom(
                name=f"tray_wall_{side}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=position,
                size=size,
                rgba=(0.45, 0.29, 0.15, 1.0),
                friction=(1.2, 0.02, 0.002),
            )

        package = world.add_body(name="package", pos=self.pickup_xyz)
        package.add_freejoint(name="package_free")
        package.add_geom(
            name="package_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self.config.package_half_size_xyz,
            mass=self.config.package_mass,
            rgba=(0.16, 0.22, 0.32, 1.0),
            friction=(1.1, 0.015, 0.002),
            condim=4,
        )

        super().__init__(model=spec.compile())
        self.package_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "package")
        self.package_joint_id = self._required_id(mujoco.mjtObj.mjOBJ_JOINT, "package_free")
        self.package_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "package_geom")
        self.table_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "contact_table_surface")
        self.tray_floor_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "tray_floor")
        self.tray_wall_geom_ids = np.asarray(
            [self._required_id(mujoco.mjtObj.mjOBJ_GEOM, f"tray_wall_{side}")
             for side in ("front", "back", "left", "right")], dtype=np.int32
        )
        self.gripper_actuator_id = self._required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.left_finger_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    def _validate_task_point(self, name: str, point: tuple[float, float, float]) -> np.ndarray:
        try:
            xyz = np.asarray(point, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must contain three finite coordinates") from error
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError(f"{name} must contain three finite coordinates")
        return xyz.copy()

    def _validate_task_geometry(self) -> None:
        config = self.config
        pickup_expected_z = config.table_surface_z + config.package_half_size_xyz[2]
        if not np.isclose(self.pickup_xyz[2], pickup_expected_z, atol=0.01):
            raise ValueError("pickup_xyz must place the package on the table within the workspace")
        if not np.isclose(self.dropoff_xyz[2], config.package_resting_z, atol=0.01):
            raise ValueError("dropoff_xyz must place the package on the tray floor within the workspace")

        table_x, table_y, _ = config.table_half_size_xyz
        table_cx, table_cy, _ = config.table_center_xyz
        package_x, package_y, _ = config.package_half_size_xyz
        tray_outer = config.tray_inner_half_size_xy + config.tray_wall_thickness
        pickup_inside = (
            abs(self.pickup_xyz[0] - table_cx) + package_x <= table_x
            and abs(self.pickup_xyz[1] - table_cy) + package_y <= table_y
        )
        tray_inside = (
            abs(self.dropoff_xyz[0] - table_cx) + tray_outer <= table_x
            and abs(self.dropoff_xyz[1] - table_cy) + tray_outer <= table_y
        )
        if not pickup_inside or not tray_inside:
            raise ValueError("pickup and dropoff must fit inside the table workspace")
        if max(abs(self.dropoff_xyz[0] - table_cx), abs(self.dropoff_xyz[1] - table_cy)) > max(table_x, table_y):
            raise ValueError("dropoff lies outside the table workspace")

    def _required_id(self, kind: mujoco.mjtObj, name: str) -> int:
        identifier = mujoco.mj_name2id(self.model, kind, name)
        if identifier < 0:
            raise ValueError(f"Contact manipulation model is missing element '{name}'")
        return identifier

    def get_package_position(self) -> np.ndarray:
        """Return a copy of the package center in world coordinates."""
        return self.data.xpos[self.package_body_id].copy()

    def get_package_linear_velocity(self) -> np.ndarray:
        """Return the free-joint translational velocity in world coordinates."""
        address = int(self.model.jnt_dofadr[self.package_joint_id])
        return self.data.qvel[address + 3:address + 6].copy()

    def package_has_gripper_contact(self) -> bool:
        """Return whether the package touches the hand or either finger."""
        hand_body = self.hand_id
        gripper_bodies = {hand_body, self.left_finger_body_id, self.right_finger_body_id}
        return self._package_contacts_body(gripper_bodies)

    def _package_contacts_body(self, body_ids: set[int]) -> bool:
        for contact in self.data.contact[:self.data.ncon]:
            if contact.dist > 0.001:
                continue
            geom_bodies = (
                int(self.model.geom_bodyid[contact.geom1]),
                int(self.model.geom_bodyid[contact.geom2]),
            )
            if self.package_geom_id in (contact.geom1, contact.geom2) and any(
                body_id in geom_bodies for body_id in body_ids
            ):
                return True
        return False

    def package_has_support_contact(self) -> bool:
        """Return whether a current contact supports the package on table or tray."""
        supports = {self.table_geom_id, self.tray_floor_geom_id}
        return any(
            {int(contact.geom1), int(contact.geom2)}.issubset({self.package_geom_id, *supports})
            and self.package_geom_id in (contact.geom1, contact.geom2)
            and any(support in (contact.geom1, contact.geom2) for support in supports)
            and contact.dist <= 0.001
            for contact in self.data.contact[:self.data.ncon]
        )

    def is_inside_tray(self, xyz: np.ndarray | None = None) -> bool:
        """Check package center is inside the usable tray footprint."""
        point = self.get_package_position() if xyz is None else np.asarray(xyz, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("package position must contain three finite coordinates")
        inner_x = self.config.tray_inner_half_size_xy - self.config.package_half_size_xyz[0]
        inner_y = self.config.tray_inner_half_size_xy - self.config.package_half_size_xyz[1]
        return bool(
            abs(point[0] - self.dropoff_xyz[0]) <= inner_x
            and abs(point[1] - self.dropoff_xyz[1]) <= inner_y
        )

    def is_package_stable_on_support(
        self, *, max_linear_speed: float = 0.02, max_angular_speed: float = 0.5
    ) -> bool:
        """Require table/tray support and low package velocity."""
        if not self.package_has_support_contact():
            return False
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.package_body_id, velocity, 0
        )
        return bool(
            np.linalg.norm(velocity[3:]) <= max_linear_speed
            and np.linalg.norm(velocity[:3]) <= max_angular_speed
        )

    def open_gripper(self) -> None:
        """Command the named Panda gripper actuator to its open setting."""
        self.data.ctrl[self.gripper_actuator_id] = self.model.actuator_ctrlrange[
            self.gripper_actuator_id, 1
        ]

    def close_gripper(self) -> None:
        """Command the named Panda gripper actuator to its closed setting."""
        self.data.ctrl[self.gripper_actuator_id] = self.model.actuator_ctrlrange[
            self.gripper_actuator_id, 0
        ]
