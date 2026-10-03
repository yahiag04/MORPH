"""Contact-based Panda pick-and-place replay driven by a human XY path."""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np

from .contact_manipulation_env import ContactManipulationEnv
from .ik import solve_position_ik


def run_demonstration_manipulation(
    env: ContactManipulationEnv,
    timed_xy: np.ndarray,
    pickup_xy: tuple[float, float],
    dropoff_xy: tuple[float, float],
    *,
    heights: dict[str, float] | None = None,
    radii: dict[str, float] | None = None,
    on_control_step: Callable[[dict], None] | None = None,
    simulation_steps_per_sample: int = 10,
) -> dict:
    """Replay timestamped Cartesian samples with inferred grasp/place phases.

    ``timed_xy`` has columns ``timestamp,x,y,z``. The recording determines XY;
    grasp heights and trigger radii are explicit assumptions for overhead video.
    Object motion always comes from MuJoCo contacts.
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
    if simulation_steps_per_sample < 1:
        raise ValueError("simulation_steps_per_sample must be positive")
    heights = heights or {"approach": 0.52, "grasp": env.config.table_surface_z + 0.035,
                          "carry": 0.56, "release": env.config.table_surface_z + 0.035}
    radii = radii or {"pickup": 0.06, "dropoff": 0.08}
    phase = "approach"
    gripper_command = "open"
    initial_package_z = float(env.get_package_position()[2])
    max_lift = 0.0
    logs: list[dict] = []
    failure = None
    saw_pickup = False
    saw_dropoff = False

    for timestamp, x, y, recorded_z in path:
        point_xy = np.array((x, y))
        if phase == "approach" and np.linalg.norm(point_xy - pickup) <= radii["pickup"]:
            phase, gripper_command, saw_pickup = "grasp", "close", True
        elif phase == "carry" and np.linalg.norm(point_xy - dropoff) <= radii["dropoff"]:
            phase, gripper_command, saw_dropoff = "release", "open", True

        if phase == "approach":
            target = np.array((x, y, heights["approach"]))
        elif phase == "grasp":
            target = np.array((pickup[0], pickup[1], heights["grasp"]))
        elif phase == "carry":
            target = np.array((x, y, heights["carry"]))
        elif phase == "release":
            target = np.array((dropoff[0], dropoff[1], heights["release"]))
        else:
            target = np.array((x, y, heights["approach"]))

        ik = solve_position_ik(env, target)
        env.close_gripper() if gripper_command == "close" else env.open_gripper()
        if not ik.converged and failure is None:
            failure = "ik_not_converged"
        for _ in range(simulation_steps_per_sample):
            env.step()
            current_z = float(env.get_package_position()[2])
            max_lift = max(max_lift, current_z - initial_package_z)
        log = {
            "timestamp": float(timestamp), "phase": phase, "target_xyz": target.copy(),
            "actual_xyz": env.get_end_effector_position().copy(), "gripper_command": gripper_command,
            "package_xyz": env.get_package_position(), "package_linear_velocity": env.get_package_linear_velocity(),
            "package_gripper_contact": env.package_has_gripper_contact(),
            "package_support_contact": env.package_has_support_contact(),
            "ik_converged": bool(ik.converged), "ik_error_m": float(ik.position_error),
        }
        logs.append(log)
        if on_control_step is not None:
            on_control_step(log)
        if phase == "grasp":
            phase = "carry"
        elif phase == "release":
            phase = "complete"
            break

    if not saw_pickup:
        failure = "pickup_trigger_not_reached"
    elif not saw_dropoff:
        failure = "dropoff_trigger_not_reached"
    elif phase != "complete" and failure is None:
        failure = "episode_incomplete"
    elif any(not row["ik_converged"] for row in logs):
        failure = "ik_not_converged"
    settled = env.is_package_stable_on_support() and env.is_inside_tray()
    success = bool(failure is None and max_lift >= 0.05 and settled)
    return {
        "phase": phase, "success": success, "failure_reason": failure,
        "max_lift_m": float(max_lift), "stable_in_tray": bool(settled),
        "ik_failures": sum(not row["ik_converged"] for row in logs), "steps": len(logs),
        "transitions": logs,
    }
