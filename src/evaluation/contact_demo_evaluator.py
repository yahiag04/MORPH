"""Evaluate recorded hand paths as physical Panda contact-manipulation trials."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from simulation.contact_manipulation_env import ContactManipulationEnv
from simulation.demonstration_manipulation import run_demonstration_manipulation
from simulation.ik import ARM_JOINT_NAMES
from simulation.panda_env import DEFAULT_MODEL_PATH

STATE_DIM = 19
ACTION_DIM = 4


def _state_vector(env: ContactManipulationEnv) -> np.ndarray:
    arm_qpos = []
    for name in ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        arm_qpos.append(env.data.qpos[env.model.jnt_qposadr[joint_id]])
    finger_qpos = []
    for name in ("finger_joint1", "finger_joint2"):
        joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        finger_qpos.append(env.data.qpos[env.model.jnt_qposadr[joint_id]])
    gripper_aperture = float(sum(finger_qpos))
    return np.concatenate((
        env.get_end_effector_position(), np.asarray(arm_qpos), env.get_package_position(),
        env.get_package_linear_velocity(), np.asarray((gripper_aperture,
            float(env.package_has_gripper_contact()), float(env.package_has_support_contact()))),
    )).astype(np.float32)


def evaluate_contact_trajectory(
    trajectory_xyz: np.ndarray,
    pickup_xy: tuple[float, float] | np.ndarray,
    dropoff_xy: tuple[float, float] | np.ndarray,
    *,
    human_label: str,
    model_path: str | Path | None = None,
    simulation_steps_per_sample: int = 10,
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
    )
    initial_state = _state_vector(env)
    after: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    def record(row: dict) -> None:
        state = _state_vector(env)
        target = np.asarray(row["target_xyz"], dtype=np.float32)
        command = 0.0 if row["gripper_command"] == "close" else 0.08
        actions.append(np.concatenate((target, np.asarray((command,), dtype=np.float32))))
        after.append(state.copy())

    result = run_demonstration_manipulation(
        env, trajectory, pickup, dropoff, on_control_step=record,
        simulation_steps_per_sample=simulation_steps_per_sample,
    )
    transitions = []
    previous = initial_state
    for action, following, log in zip(actions, after, result["transitions"], strict=True):
        transitions.append({"state": previous, "action": action, "next_state": following,
                            "phase": log["phase"], "timestamp": log["timestamp"]})
        previous = following
    return {
        "human_label": str(human_label), "robot_success": bool(result["success"]),
        "failure_reason": result["failure_reason"], "max_lift_m": result["max_lift_m"],
        "stable_in_tray": result["stable_in_tray"], "ik_failures": result["ik_failures"],
        "steps": result["steps"], "transitions": transitions,
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
    for run in runs:
        for transition in run.get("transitions", []):
            states.append(transition["state"])
            actions.append(transition["action"])
            next_states.append(transition["next_state"])
            clip_ids.append(str(run["clip_id"]))
            labels.append(str(run["human_label"]))
            phases.append(str(transition["phase"]))
    np.savez_compressed(
        Path(destination), states=np.asarray(states, dtype=np.float32).reshape(-1, STATE_DIM),
        actions=np.asarray(actions, dtype=np.float32).reshape(-1, ACTION_DIM),
        next_states=np.asarray(next_states, dtype=np.float32).reshape(-1, STATE_DIM),
        clip_ids=np.asarray(clip_ids), human_labels=np.asarray(labels), phases=np.asarray(phases),
    )
