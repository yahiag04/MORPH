"""Run a Panda Cartesian trajectory and save its measured tracking results."""

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from evaluation.trajectory_plot import plot_trajectory
from simulation.panda_env import PandaEnv
from simulation.trajectory_player import CartesianTrajectoryPlayer


SEED = np.array([0.0, -0.5, 0.0, -1.7, 0.0, 2.0, 0.0])
POSES = (
    SEED,
    np.array([0.12, -0.42, 0.12, -1.82, 0.08, 2.1, 0.1]),
    np.array([0.22, -0.35, 0.2, -1.9, 0.15, 2.15, 0.2]),
)


def main() -> None:
    """Run the viewer demo and write CSV and PNG measurements."""
    root = Path(__file__).resolve().parents[1]
    env = PandaEnv()
    env.data.qpos[:7] = SEED
    env.data.ctrl[:7] = SEED
    mujoco.mj_forward(env.model, env.data)

    waypoints = []
    for pose in POSES:
        env.data.qpos[:7] = pose
        mujoco.mj_forward(env.model, env.data)
        waypoints.append(env.get_end_effector_position())
    env.data.qpos[:7] = SEED
    env.data.ctrl[:7] = SEED
    mujoco.mj_forward(env.model, env.data)

    player = CartesianTrajectoryPlayer(env)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        log = player.follow(
            np.asarray(waypoints),
            output_csv=root / "results/metrics/trajectory.csv",
            on_control_step=lambda: (viewer.sync(), time.sleep(player.control_period)),
        )

    figure_path = plot_trajectory(
        log, root / "results/figures/trajectory_tracking.png"
    )
    print(f"CSV: {root / 'results/metrics/trajectory.csv'}")
    print(f"Figure: {figure_path}")
    print(f"Mean position error: {log.position_error.mean():.6f} m")
    print(f"Maximum position error: {log.position_error.max():.6f} m")
    print(
        "Simulation samples with unconverged IK: "
        f"{np.count_nonzero(~log.ik_converged)}"
    )


if __name__ == "__main__":
    main()
