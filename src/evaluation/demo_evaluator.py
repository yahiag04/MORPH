"""Deterministic headless Panda trajectory evaluation."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from simulation.panda_env import PandaEnv
from simulation.trajectory_player import CartesianTrajectoryPlayer, TrajectoryLog


REPLAY_SEED = np.asarray((0.0, -0.5, 0.0, -1.7, 0.0, 2.0, 0.0), dtype=np.float64)


def evaluate_trajectory(
    trajectory: np.ndarray,
    model_path: str | Path | None = None,
) -> tuple[TrajectoryLog, dict[str, float | int]]:
    """Replay one finite Panda XYZ path from the fixed demo seed without a viewer."""
    try:
        points = np.asarray(trajectory, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("trajectory must contain numeric XYZ coordinates") from error
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise ValueError("trajectory must contain at least two XYZ waypoints")
    if not np.isfinite(points).all():
        raise ValueError("trajectory coordinates must all be finite")

    env = PandaEnv() if model_path is None else PandaEnv(model_path)
    env.data.qpos[: REPLAY_SEED.size] = REPLAY_SEED
    env.data.ctrl[: REPLAY_SEED.size] = REPLAY_SEED
    mujoco.mj_forward(env.model, env.data)
    player = CartesianTrajectoryPlayer(env)
    log = player.follow(points)
    metrics: dict[str, float | int] = {
        "sample_count": int(log.timestamps.size),
        "mean_position_error_m": float(log.position_error.mean()),
        "max_position_error_m": float(log.position_error.max()),
        "ik_nonconverged_count": int(np.count_nonzero(~log.ik_converged)),
    }
    return log, metrics
