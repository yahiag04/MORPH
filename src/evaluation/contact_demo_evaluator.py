"""Evaluate recorded hand paths as physical Panda contact-manipulation trials."""

from __future__ import annotations

import json
from collections import Counter
from copy import copy
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np

from simulation.contact_manipulation_env import ContactManipulationEnv, ContactTaskConfig
from simulation.demonstration_manipulation import run_demonstration_manipulation
from simulation.ik import ARM_JOINT_NAMES
from simulation.panda_env import DEFAULT_MODEL_PATH

STATE_DIM = 37
ACTION_DIM = 4


def sample_contact_control_rate(trajectory: np.ndarray, hz: float) -> np.ndarray:
    """Match a timestamped hand path to the task controller's command rate."""
    points = np.asarray(trajectory, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 4 or len(points) < 1:
        raise ValueError("trajectory must have columns timestamp,x,y,z")
    if not np.isfinite(points).all() or not np.isfinite(hz) or hz <= 0:
        raise ValueError("trajectory and control frequency must be finite; frequency must be positive")
    interval = 1.0 / hz
    indices = [0]
    last_time = float(points[0, 0])
    for index in range(1, len(points) - 1):
        if points[index, 0] - last_time >= interval:
            indices.append(index)
            last_time = float(points[index, 0])
    if len(points) > 1 and indices[-1] != len(points) - 1:
        indices.append(len(points) - 1)
    sampled = points[np.asarray(indices)]
    if len(sampled) > 1 and np.any(np.diff(sampled[:, 0]) <= 0):
        raise ValueError("downsampled trajectory timestamps are not strictly increasing")
    return sampled


def contact_task_state(env: ContactManipulationEnv, *, observation_data: mujoco.MjData | None = None) -> np.ndarray:
    """Read a coherent 37-value observation without touching the live solver.

    After mj_step, derived positions and contacts still describe the preceding
    physics step. Refresh them on a snapshot at the current qpos/qvel/time.
    Callers collecting many observations can reuse the scratch MjData.
    """
    if observation_data is env.data:
        raise ValueError('observation_data must be separate from live simulation data')
    snapshot = copy(env)
    snapshot.data = observation_data if observation_data is not None else mujoco.MjData(env.model)
    mujoco.mj_copyData(snapshot.data, env.model, env.data)
    mujoco.mj_forward(env.model, snapshot.data)
    env = snapshot
    arm_qpos = []
    arm_qvel = []
    for name in ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        arm_qpos.append(env.data.qpos[env.model.jnt_qposadr[joint_id]])
        arm_qvel.append(env.data.qvel[env.model.jnt_dofadr[joint_id]])
    finger_qpos = []
    finger_qvel = []
    for name in ("finger_joint1", "finger_joint2"):
        joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        finger_qpos.append(env.data.qpos[env.model.jnt_qposadr[joint_id]])
        finger_qvel.append(env.data.qvel[env.model.jnt_dofadr[joint_id]])
    gripper_aperture = float(sum(finger_qpos))
    package_qpos = int(env.model.jnt_qposadr[env.package_joint_id])
    package_dof = int(env.model.jnt_dofadr[env.package_joint_id])
    return np.concatenate((
        env.get_end_effector_position(), np.asarray(arm_qpos), np.asarray(arm_qvel),
        env.get_package_position(), env.data.qpos[package_qpos + 3:package_qpos + 7],
        env.data.qvel[package_dof:package_dof + 3], env.data.qvel[package_dof + 3:package_dof + 6],
        np.asarray((gripper_aperture, sum(finger_qvel),
            float(env.package_has_gripper_contact()), float(env.package_has_support_contact()))),
        env.dropoff_xyz,
    )).astype(np.float32)


def evaluate_contact_trajectory(
    trajectory_xyz: np.ndarray,
    pickup_xy: tuple[float, float] | np.ndarray,
    dropoff_xy: tuple[float, float] | np.ndarray,
    *,
    human_label: str,
    model_path: str | Path | None = None,
    simulation_steps_per_sample: int = 10,
    observation_steps: int | None = None,
    config: ContactTaskConfig | None = None,
    heights: dict[str, float] | None = None,
    cartesian_speed_mps: float = 0.08,
    release_open_fraction: float = 0.0,
    controller_pickup_xy: np.ndarray | None = None,
    controller_dropoff_xy: np.ndarray | None = None,
    on_observation: Callable[[ContactManipulationEnv, dict | None], None] | None = None,
) -> dict:
    """Run one fresh simulation and return outcome plus state/action transitions."""
    trajectory = np.asarray(trajectory_xyz, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 4 or len(trajectory) == 0:
        raise ValueError("trajectory must have columns timestamp,x,y,z")
    model = Path(model_path).expanduser() if model_path is not None else DEFAULT_MODEL_PATH
    pickup = np.asarray(pickup_xy, dtype=np.float64)
    dropoff = np.asarray(dropoff_xy, dtype=np.float64)
    env = ContactManipulationEnv(
        model_path=model,
        pickup_xyz=(float(pickup[0]), float(pickup[1]), 0.412),
        dropoff_xyz=(float(dropoff[0]), float(dropoff[1]), 0.416),
        config=config,
    )
    observation_data = mujoco.MjData(env.model)
    initial_state = contact_task_state(env, observation_data=observation_data)
    transitions = []
    previous = initial_state
    if on_observation is not None:
        on_observation(env, None)

    def record(row: dict) -> None:
        nonlocal previous
        state = contact_task_state(env, observation_data=observation_data)
        target = np.asarray(row["target_xyz"], dtype=np.float32)
        command = 0.0 if row["gripper_command"] == "close" else 0.08
        transitions.append({
            "state": previous, "next_state": state.copy(),
            "action": np.concatenate((target, np.asarray((command,), dtype=np.float32))),
            "actuator_controls": row["actuator_controls"].copy(),
            "simulation_start_time": row["simulation_start_time"],
            "simulation_end_time": row["simulation_end_time"],
            "dt_seconds": row["simulation_end_time"] - row["simulation_start_time"],
            "timestamp": row["simulation_end_time"], "human_timestamp": row["human_timestamp"],
            "phase": row["phase"],
        })
        previous = state.copy()
        if on_observation is not None:
            on_observation(env, row)

    result = run_demonstration_manipulation(
        env, trajectory,
        pickup if controller_pickup_xy is None else controller_pickup_xy,
        dropoff if controller_dropoff_xy is None else controller_dropoff_xy,
        on_observation=record, observation_steps=observation_steps,
        simulation_steps_per_sample=simulation_steps_per_sample,
        heights=heights, cartesian_speed_mps=cartesian_speed_mps,
        release_open_fraction=release_open_fraction,
    )
    return {
        "human_label": str(human_label), "robot_success": bool(result["success"]),
        "failure_reason": result["failure_reason"], "max_lift_m": result["max_lift_m"],
        "stable_in_tray": result["stable_in_tray"], "ik_failures": result["ik_failures"],
        "steps": len(transitions), "control_steps": result["steps"], "transitions": transitions,
        "physics_timestep_seconds": float(env.model.opt.timestep),
        "control_dt_seconds": float(env.model.opt.timestep * simulation_steps_per_sample),
        "observation_dt_seconds": float(env.model.opt.timestep * (observation_steps or simulation_steps_per_sample)),
        "final_package_xyz": env.get_package_position(),
    }


def aggregate_contact_runs(runs: list[dict]) -> dict:
    """Aggregate every attempted episode, preserving separate human and robot labels."""
    label_counts = Counter(str(run["human_label"]) for run in runs)
    by_label: dict[str, dict] = {}
    for label in sorted(label_counts):
        subset = [run for run in runs if str(run["human_label"]) == label]
        success_count = sum(bool(run["robot_success"]) for run in subset)
        by_label[label] = {
            "episodes": len(subset), "robot_successes": success_count,
            "robot_failures": len(subset) - success_count,
            "robot_success_rate": success_count / len(subset) if subset else None,
        }
    failure_reasons = Counter(str(run["failure_reason"]) for run in runs if run.get("failure_reason"))
    human_success = {"riusciti", "successful", "success"}
    human_failure = {"falliti", "failed", "failure"}
    confusion = {
        "human_success_robot_success": sum(bool(r["robot_success"]) and str(r["human_label"]).lower() in human_success for r in runs),
        "human_success_robot_failure": sum(not bool(r["robot_success"]) and str(r["human_label"]).lower() in human_success for r in runs),
        "human_failure_robot_success": sum(bool(r["robot_success"]) and str(r["human_label"]).lower() in human_failure for r in runs),
        "human_failure_robot_failure": sum(not bool(r["robot_success"]) and str(r["human_label"]).lower() in human_failure for r in runs),
    }
    successes = sum(bool(run["robot_success"]) for run in runs)
    values = [float(run["max_lift_m"]) for run in runs if run.get("max_lift_m") is not None]
    return {
        "episodes": len(runs), "robot_successes": successes,
        "robot_success_rate": successes / len(runs) if runs else None,
        "human_label_counts": dict(sorted(label_counts.items())), "by_human_label": by_label,
        "confusion": confusion, "failure_reasons": dict(sorted(failure_reasons.items())),
        "mean_max_lift_m": float(np.mean(values)) if values else None,
        "mean_transition_count": float(np.mean([int(r.get("steps", 0)) for r in runs])) if runs else None,
        "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
    }


def write_transition_archive(runs: list[dict], destination: str | Path) -> None:
    """Save local transitions with clip identifiers and labels for clip-held-out training."""
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    clip_ids: list[str] = []
    labels: list[str] = []
    phases: list[str] = []
    controls, starts, ends, intervals = [], [], [], []
    episodes, groups, sources, splits = [], [], [], []
    for run in runs:
        for transition in run.get("transitions", []):
            states.append(transition["state"])
            actions.append(transition["action"])
            next_states.append(transition["next_state"])
            clip_ids.append(str(run["clip_id"]))
            labels.append(str(run["human_label"]))
            phases.append(str(transition["phase"]))
            controls.append(transition["actuator_controls"])
            starts.append(transition["simulation_start_time"])
            ends.append(transition["simulation_end_time"])
            intervals.append(transition["dt_seconds"])
            episodes.append(str(run.get("episode_id", run["clip_id"])))
            groups.append(str(run.get("group_id", run["clip_id"])))
            sources.append(str(run.get("source_kind", "human_replay")))
            splits.append(str(run.get("split", "unassigned")))
    np.savez_compressed(
        Path(destination), states=np.asarray(states, dtype=np.float32).reshape(-1, STATE_DIM),
        actions=np.asarray(actions, dtype=np.float32).reshape(-1, ACTION_DIM),
        next_states=np.asarray(next_states, dtype=np.float32).reshape(-1, STATE_DIM),
        clip_ids=np.asarray(clip_ids), human_labels=np.asarray(labels), phases=np.asarray(phases),
        actuator_controls=np.asarray(controls, dtype=np.float64).reshape(-1, 8),
        simulation_start_times=np.asarray(starts, dtype=np.float64),
        simulation_end_times=np.asarray(ends, dtype=np.float64),
        transition_dt_seconds=np.asarray(intervals, dtype=np.float64),
        episode_ids=np.asarray(episodes, dtype=str), group_ids=np.asarray(groups, dtype=str),
        source_kinds=np.asarray(sources, dtype=str), splits=np.asarray(splits, dtype=str),
        archive_schema_version=np.asarray(3, dtype=np.int32),
    )
