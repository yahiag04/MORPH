"""Oracle pick-and-place sequence for the simple Panda cube task."""

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from .manipulation_env import ManipulationEnv
from .trajectory_player import CartesianTrajectoryPlayer, TrajectoryLog


@dataclass(frozen=True)
class PickPlaceResult:
    """Measured outcome of one scripted cube pick-and-place attempt."""

    log: TrajectoryLog
    initial_cube_xyz: np.ndarray
    final_cube_xyz: np.ndarray
    max_cube_z: float
    object_lifted: bool
    object_placed: bool
    success: bool


def run_oracle_pick_and_place(
    env: ManipulationEnv,
    *,
    speed: float = 0.08,
    control_period: float = 0.02,
    on_control_step: Callable[[], None] | None = None,
) -> PickPlaceResult:
    """Pick the configured cube and attempt to place it inside the target.

    The sequence uses only arm position actuators, the Panda gripper actuator,
    and normal MuJoCo contact dynamics. The cube is never teleported or
    attached to the hand.
    """
    player = CartesianTrajectoryPlayer(
        env, speed=speed, control_period=control_period, smoothing_window=1
    )
    initial_cube_xyz = env.get_cube_position()
    max_cube_z = float(initial_cube_xyz[2])
    logs: list[TrajectoryLog] = []

    def track_cube() -> None:
        nonlocal max_cube_z
        max_cube_z = max(max_cube_z, float(env.get_cube_position()[2]))

    def observe() -> None:
        track_cube()
        if on_control_step is not None:
            on_control_step()

    def hold(seconds: float) -> None:
        step_count = int(round(seconds / player.simulation_timestep))
        for index in range(step_count):
            env.step()
            mujoco.mj_forward(env.model, env.data)
            track_cube()
            is_control_boundary = (index + 1) % player.steps_per_control == 0
            if on_control_step is not None and (
                is_control_boundary or index + 1 == step_count
            ):
                on_control_step()

    def move_to(xyz: np.ndarray) -> None:
        current = env.get_end_effector_position()
        logs.append(
            player.follow(
                np.vstack((current, xyz)), on_control_step=observe
            )
        )

    cube_start = np.asarray(env.config.cube_start_xyz, dtype=np.float64)
    target = np.asarray(env.config.target_xyz, dtype=np.float64)
    gripper_offset = np.asarray(
        env.config.gripper_center_offset_xy, dtype=np.float64
    )
    surface_z = env.config.table_surface_z
    approach_height = surface_z + 0.26
    grasp_height = surface_z + 0.12
    lift_height = surface_z + 0.30
    env.open_gripper()
    hold(0.3)
    move_to(
        np.array([cube_start[0], cube_start[1], approach_height])
        - np.array([gripper_offset[0], gripper_offset[1], 0.0])
    )
    move_to(
        np.array([cube_start[0], cube_start[1], grasp_height])
        - np.array([gripper_offset[0], gripper_offset[1], 0.0])
    )
    env.close_gripper()
    hold(0.6)
    move_to(
        np.array([cube_start[0], cube_start[1], lift_height])
        - np.array([gripper_offset[0], gripper_offset[1], 0.0])
    )
    move_to(
        np.array([target[0], target[1], lift_height])
        - np.array([gripper_offset[0], gripper_offset[1], 0.0])
    )
    move_to(
        np.array([target[0], target[1], grasp_height])
        - np.array([gripper_offset[0], gripper_offset[1], 0.0])
    )
    env.open_gripper()
    hold(0.6)
    move_to(
        np.array([target[0], target[1], approach_height])
        - np.array([gripper_offset[0], gripper_offset[1], 0.0])
    )
    hold(0.2)

    log = TrajectoryLog(
        timestamps=np.concatenate([item.timestamps for item in logs]),
        target_xyz=np.concatenate([item.target_xyz for item in logs]),
        actual_xyz=np.concatenate([item.actual_xyz for item in logs]),
        ik_converged=np.concatenate([item.ik_converged for item in logs]),
    )
    final_cube_xyz = env.get_cube_position()
    lifted = max_cube_z - initial_cube_xyz[2] >= 0.05
    resting_height = surface_z + env.config.cube_half_size
    placed = (
        env.is_in_target_region(final_cube_xyz)
        and abs(final_cube_xyz[2] - resting_height) <= 0.03
        and env.is_cube_stably_on_table()
    )
    success = bool(lifted and placed and log.ik_converged.all())
    return PickPlaceResult(
        log=log,
        initial_cube_xyz=initial_cube_xyz,
        final_cube_xyz=final_cube_xyz,
        max_cube_z=max_cube_z,
        object_lifted=bool(lifted),
        object_placed=bool(placed),
        success=success,
    )
