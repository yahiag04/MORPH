"""Replay a mapped human trajectory on the Panda model and save measurements."""

import argparse
import csv
import json
from pathlib import Path
import re
import time

import mujoco
import mujoco.viewer
import numpy as np

from evaluation.trajectory_plot import plot_trajectory
from simulation.panda_env import PandaEnv
from simulation.trajectory_player import CartesianTrajectoryPlayer


SEED = np.array([0.0, -0.5, 0.0, -1.7, 0.0, 2.0, 0.0])


def read_robot_trajectory(path: str | Path) -> np.ndarray:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    points = np.asarray([[float(row[axis]) for axis in ("x", "y", "z")] for row in rows], dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("trajectory CSV must contain at least two finite XYZ rows")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_csv")
    parser.add_argument("--results-name", default="human_demo", help="Name for local results files")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.results_name):
        parser.error("--results-name may contain only letters, numbers, '_' and '-'")
    root = Path(__file__).resolve().parents[1]
    trajectory = read_robot_trajectory(args.trajectory_csv)
    env = PandaEnv()
    env.data.qpos[:7] = SEED
    env.data.ctrl[:7] = SEED
    mujoco.mj_forward(env.model, env.data)
    player = CartesianTrajectoryPlayer(env)
    metrics_dir = root / "results/metrics"
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        log = player.follow(trajectory, output_csv=metrics_dir / f"{args.results_name}.csv", on_control_step=lambda: (viewer.sync(), time.sleep(player.control_period)))
    figure_path = plot_trajectory(log, root / "results/figures" / f"{args.results_name}.png")
    metrics = {
        "source_trajectory": str(Path(args.trajectory_csv).expanduser().resolve()),
        "sample_count": int(log.timestamps.size),
        "mean_position_error_m": float(log.position_error.mean()),
        "max_position_error_m": float(log.position_error.max()),
        "unconverged_ik_samples": int(np.count_nonzero(~log.ik_converged)),
    }
    metrics_path = metrics_dir / f"{args.results_name}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Mean position error: {log.position_error.mean():.6f} m")
    print(f"Maximum position error: {log.position_error.max():.6f} m")
    print(f"Figure: {figure_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
