"""Contact-based Panda pick-and-place replay driven by a human XY path."""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np

from .contact_manipulation_env import ContactManipulationEnv
from .ik import solve_pose_ik, solve_position_ik


def run_demonstration_manipulation(
    env: ContactManipulationEnv,
    timed_xy: np.ndarray,
    pickup_xy: tuple[float, float],
    dropoff_xy: tuple[float, float],
    *,
    heights: dict[str, float] | None = None,
    radii: dict[str, float] | None = None,
    on_control_step: Callable[[dict], None] | None = None,
    on_observation: Callable[[dict], None] | None = None,
    observation_steps: int | None = None,
    simulation_steps_per_sample: int = 10,
    cartesian_speed_mps: float = 0.08,
    release_open_fraction: float = 0.0,
) -> dict:
    """Replay timestamped Cartesian samples with inferred grasp/place phases.

    ``timed_xy`` has columns ``timestamp,x,y,z``. The recording determines XY;
    grasp heights and trigger radii are explicit assumptions for overhead video.
    Arm targets are interpolated and followed by Panda position actuators; the
    object remains a free body and moves only through MuJoCo contact dynamics.
    """
    path = np.asarray(timed_xy, dtype=np.float64)
    pickup = np.asarray(pickup_xy, dtype=np.float64)
    dropoff = np.asarray(dropoff_xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 4 or len(path) < 1 or not np.isfinite(path).all():
        raise ValueError("timed_xy must be a finite non-empty array with columns timestamp,x,y,z")
    if np.any(np.diff(path[:, 0]) <= 0):
        raise ValueError("trajectory timestamps must be strictly increasing")
    if pickup.shape != (2,) or dropoff.shape != (2,) or not np.isfinite(pickup).all() or not np.isfinite(dropoff).all():
        raise ValueError("pickup_xy and dropoff_xy must be finite XY points")
    if not isinstance(simulation_steps_per_sample, (int, np.integer)) or simulation_steps_per_sample < 1:
        raise ValueError("simulation_steps_per_sample must be positive")
    observation_steps = simulation_steps_per_sample if observation_steps is None else observation_steps
    if (not isinstance(observation_steps, (int, np.integer)) or observation_steps < 1
            or simulation_steps_per_sample % observation_steps):
        raise ValueError("observation_steps must be a positive integer and divide simulation_steps_per_sample")
    if not np.isfinite(release_open_fraction) or not 0 <= release_open_fraction <= 1:
        raise ValueError("release_open_fraction must be between zero and one")
    if not np.isfinite(cartesian_speed_mps) or cartesian_speed_mps <= 0:
        raise ValueError("cartesian_speed_mps must be finite and positive")

    heights = heights or {"approach": 0.62, "grasp": env.config.table_surface_z + 0.12,
                          "carry": 0.70, "release": env.config.table_surface_z + 0.12}
    radii = radii or {"pickup": 0.06, "dropoff": 0.08}
    phase = "approach"
    gripper_command = "open"
    initial_package_z = float(env.get_package_position()[2])
    max_lift = 0.0
    package_ee_offset: np.ndarray | None = None
    grasp_orientation: np.ndarray | None = None
    logs: list[dict] = []
    failure = None
    saw_pickup = False
    saw_dropoff = False
    last_human_timestamp = float(path[0, 0])

    def command_arm(target_xyz: np.ndarray, orientation: np.ndarray | None = None) -> tuple[bool, float]:
        qpos_before_ik = env.data.qpos.copy()
        ik = (solve_pose_ik(env, target_xyz, orientation, position_tolerance=0.005,
                            orientation_tolerance=0.04, max_iterations=400)
              if orientation is not None else
              solve_position_ik(env, target_xyz, tolerance=0.005, max_iterations=400))
        arm_target = ik.q.copy()
        env.data.qpos[:] = qpos_before_ik
        mujoco.mj_forward(env.model, env.data)
        for index in range(7):
            actuator_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{index + 1}")
            env.data.ctrl[actuator_id] = arm_target[index]
        return bool(ik.converged), float(ik.position_error)

    def advance(target_xyz, control_phase, command, timestamp, converged, ik_error,
                *, track_offset=True):
        nonlocal max_lift, failure, package_ee_offset
        observation_start = float(env.data.time)
        controls = env.data.ctrl.copy()
        for substep in range(1, simulation_steps_per_sample + 1):
            env.step()
            current_package = env.get_package_position()
            current_z = float(current_package[2])
            max_lift = max(max_lift, current_z - initial_package_z)
            if track_offset and package_ee_offset is None and current_z - initial_package_z >= 0.05:
                package_ee_offset = current_package[:2] - env.get_end_effector_position()[:2]
            if substep % observation_steps == 0:
                if on_observation is not None:
                    on_observation({
                        "simulation_start_time": observation_start,
                        "simulation_end_time": float(env.data.time),
                        "human_timestamp": float(timestamp), "phase": control_phase,
                        "target_xyz": target_xyz.copy(), "gripper_command": command,
                        "actuator_controls": controls.copy(),
                        "ik_converged": converged, "ik_error_m": ik_error,
                    })
                observation_start = float(env.data.time)

    def control_to(target_xyz: np.ndarray, control_phase: str, command: str, timestamp: float) -> None:
        nonlocal failure
        start = env.get_end_effector_position()
        distance = float(np.linalg.norm(target_xyz - start))
        control_dt = float(env.model.opt.timestep) * simulation_steps_per_sample
        intervals = max(1, int(np.ceil(distance / (cartesian_speed_mps * control_dt))))
        env.close_gripper() if command == "close" else env.open_gripper()
        for index in range(1, intervals + 1):
            interval_command = command
            if control_phase == "release" and release_open_fraction > 0:
                interval_command = "open" if index / intervals >= release_open_fraction else "close"
                env.open_gripper() if interval_command == "open" else env.close_gripper()
            interpolated = start + (target_xyz - start) * (index / intervals)
            converged, ik_error = command_arm(interpolated)
            if not converged and failure is None:
                failure = "ik_not_converged"
            advance(interpolated, control_phase, interval_command, timestamp, converged, ik_error)
            log = {
                "timestamp": timestamp + index * control_dt,
                "phase": control_phase,
                "target_xyz": interpolated.copy(),
                "actual_xyz": env.get_end_effector_position().copy(),
                "gripper_command": interval_command,
                "package_xyz": env.get_package_position(),
                "package_linear_velocity": env.get_package_linear_velocity(),
                "package_gripper_contact": env.package_has_gripper_contact(),
                "package_support_contact": env.package_has_support_contact(),
                "ik_converged": converged,
                "ik_error_m": ik_error,
            }
            logs.append(log)
            if on_control_step is not None:
                on_control_step(log)

    def object_target(xy: np.ndarray, z: float) -> np.ndarray:
        target = np.array((xy[0], xy[1], z), dtype=np.float64)
        target[:2] -= np.asarray(env.config.gripper_center_offset_xy, dtype=np.float64)
        return target

    for timestamp, x, y, _recorded_z in path:
        point_xy = np.array((x, y), dtype=np.float64)
        if phase == "approach" and np.linalg.norm(point_xy - pickup) <= radii["pickup"]:
            saw_pickup = True
            phase = "grasp"
            pickup_approach = object_target(pickup, heights["approach"])
            control_to(pickup_approach, "approach", "open", float(timestamp))
            grasp_target = object_target(pickup, heights["grasp"])
            control_to(grasp_target, "grasp", "open", float(timestamp))
            for _ in range(3):
                control_to(grasp_target, "grasp", "close", float(timestamp))
            grasp_orientation = env.data.xquat[env.hand_id].copy()
            phase = "carry"
            # Lift straight up before following the recorded lateral path.
            control_to(object_target(pickup, heights["carry"]), "carry", "close", float(timestamp))
            last_human_timestamp = float(timestamp)
            continue

        if phase == "carry" and np.linalg.norm(point_xy - dropoff) <= radii["dropoff"]:
            saw_dropoff = True
            phase = "release"
            if env.get_package_position()[2] > initial_package_z + 0.02:
                package_ee_offset = env.get_package_position()[:2] - env.get_end_effector_position()[:2]
            if package_ee_offset is None:
                release_xy = dropoff - np.asarray(env.config.gripper_center_offset_xy, dtype=np.float64)
            else:
                release_xy = dropoff - package_ee_offset
            release_target = np.array((release_xy[0], release_xy[1], heights["release"]), dtype=np.float64)
            control_to(release_target, "release", "open", float(timestamp))
            for _ in range(max(10, int(round(0.4 / (env.model.opt.timestep * simulation_steps_per_sample))))):
                advance(release_target, "settle", "open", timestamp,
                        logs[-1]["ik_converged"], logs[-1]["ik_error_m"], track_offset=False)
            phase = "complete"
            break

        if phase == "approach":
            target = np.array((x, y, heights["approach"]), dtype=np.float64)
            command = "open"
        elif phase == "carry":
            target = np.array((x, y, heights["carry"]), dtype=np.float64)
            command = "close"
        else:
            break
        control_to(target, phase, command, float(timestamp))
        last_human_timestamp = float(timestamp)

    settled = env.is_package_stable_on_support() and env.is_inside_tray()
    if not saw_pickup:
        failure = "pickup_trigger_not_reached"
    elif not saw_dropoff:
        failure = "dropoff_trigger_not_reached"
    elif phase != "complete" and failure is None:
        failure = "episode_incomplete"
    elif any(not row["ik_converged"] for row in logs):
        failure = "ik_not_converged"
    elif max_lift < 0.05:
        failure = "package_not_lifted"
    elif not settled:
        failure = "package_not_stably_placed_in_tray"

    success = bool(failure is None and max_lift >= 0.05 and settled)
    return {
        "phase": phase, "success": success, "failure_reason": failure,
        "max_lift_m": float(max_lift), "stable_in_tray": bool(settled),
        "ik_failures": sum(not row["ik_converged"] for row in logs), "steps": len(logs),
        "transitions": logs,
    }
