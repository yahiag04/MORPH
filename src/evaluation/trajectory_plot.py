"""Plot measured Cartesian trajectory tracking results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from simulation.trajectory_player import TrajectoryLog


def plot_trajectory(
    log: TrajectoryLog, output_path: str | Path
) -> Path:
    """Save target/actual paths and tracking error as a PNG figure."""
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(11, 5))
    path_axis = figure.add_subplot(1, 2, 1, projection="3d")
    path_axis.plot(*log.target_xyz.T, label="Target", linewidth=2)
    path_axis.plot(*log.actual_xyz.T, label="Actual", linewidth=1.5)
    path_axis.set_xlabel("X (m)")
    path_axis.set_ylabel("Y (m)")
    path_axis.set_zlabel("Z (m)")
    path_axis.set_title("End-effector path")
    path_axis.legend()

    error_axis = figure.add_subplot(1, 2, 2)
    error_axis.plot(log.timestamps, log.position_error)
    error_axis.set_xlabel("Simulation time (s)")
    error_axis.set_ylabel("Position error (m)")
    error_axis.set_title("Tracking error")
    error_axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
