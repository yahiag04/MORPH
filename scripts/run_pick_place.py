"""Run the Panda cube pick-and-place demonstration in MuJoCo's viewer."""

import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from evaluation.trajectory_plot import plot_trajectory
from simulation.manipulation_env import ManipulationEnv
from simulation.pick_place import run_oracle_pick_and_place


SEED = np.array([0.0, -0.5, 0.0, -1.7, 0.0, 2.0, 0.0])


def main() -> None:
    """Execute the oracle sequence and save measured outputs."""
    root = Path(__file__).resolve().parents[1]
    env = ManipulationEnv()
    env.data.qpos[:7] = SEED
    env.data.ctrl[:7] = SEED
    env.open_gripper()
    mujoco.mj_forward(env.model, env.data)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        result = run_oracle_pick_and_place(
            env,
            on_control_step=lambda: (
                viewer.sync() if viewer.is_running() else None,
                time.sleep(0.02),
            ),
        )

    metrics = {
        "success": result.success,
        "object_lifted": result.object_lifted,
        "object_placed": result.object_placed,
        "initial_cube_xyz_m": result.initial_cube_xyz.tolist(),
        "max_cube_z_m": result.max_cube_z,
        "final_cube_xyz_m": result.final_cube_xyz.tolist(),
        "target_center_xyz_m": list(env.config.target_xyz),
        "trajectory_samples": int(result.log.timestamps.size),
        "ik_unconverged_samples": int(
            np.count_nonzero(~result.log.ik_converged)
        ),
    }
    metrics_dir = root / "results/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    result.log.write_csv(metrics_dir / "pick_place.csv")
    (metrics_dir / "pick_place.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    plot_trajectory(
        result.log, root / "results/figures/pick_place_trajectory.png"
    )
    print(json.dumps(metrics, indent=2))
    print(f"CSV: {metrics_dir / 'pick_place.csv'}")
    print(f"Figure: {root / 'results/figures/pick_place_trajectory.png'}")


if __name__ == "__main__":
    main()
