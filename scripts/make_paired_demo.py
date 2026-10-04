"""Render one Panda trajectory and pair it with a tracked real video locally."""

import argparse
import csv
from pathlib import Path

import numpy as np

from evaluation.demo_video import (
    PUBLIC_SHOWCASE_PATH, make_paired_video, render_contact_panda_replay,
    render_panda_trajectory,
)
from perception.retargeting import WorkspaceMapping
from perception.task_layout import TaskLayout


def _reject_output_input_aliases(output_paths: list[Path], inputs: list[Path]) -> None:
    resolved_outputs = [path.expanduser().resolve() for path in output_paths]
    resolved_inputs = [path.expanduser().resolve() for path in inputs]
    all_paths = resolved_outputs + resolved_inputs
    for index, path in enumerate(resolved_outputs):
        for other in all_paths[index + 1:]:
            if path == other:
                raise ValueError("generated video paths must not overwrite an input")
            try:
                if path.exists() and other.exists() and path.samefile(other):
                    raise ValueError("generated video paths must not overwrite an input")
            except OSError:
                continue


def read_trajectory(path: Path) -> np.ndarray:
    """Read finite XYZ columns from a retargeted CSV."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    try:
        points = np.asarray([[float(row[axis]) for axis in ("x", "y", "z")] for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"trajectory CSV must contain numeric x, y, z columns: {path}") from error
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("trajectory CSV must contain at least two finite XYZ rows")
    return points


def read_timed_trajectory(path: Path) -> np.ndarray:
    """Read timestamp and XYZ columns needed to replay the contact controller."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    try:
        points = np.asarray([[float(row[axis]) for axis in ("timestamp", "x", "y", "z")]
                             for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"trajectory CSV must contain numeric timestamp, x, y, z columns: {path}") from error
    if (points.ndim != 2 or len(points) < 2 or not np.isfinite(points).all()
            or np.any(np.diff(points[:, 0]) <= 0)):
        raise ValueError("trajectory CSV must contain at least two finite, time-ordered waypoints")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracked_video", type=Path, help="Raw or annotated human demonstration video")
    parser.add_argument("trajectory_csv", type=Path, help="Retargeted XYZ CSV from the evaluator")
    parser.add_argument("--output-dir", required=True, type=Path, help="Local directory outside the repository")
    parser.add_argument("--model-path", type=Path, help="Optional Panda scene XML path")
    parser.add_argument("--task-layout", type=Path,
                        help="Local task_layout.json; enables the package-and-tray contact replay")
    parser.add_argument("--human-label", default="riusciti",
                        help="Local source label used by the contact replay (default: riusciti)")
    parser.add_argument("--robot-bounds", nargs=4, type=float, default=(0.38, 0.68, -0.22, 0.22),
                        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
                        help="Workspace bounds used to map normalized task-layout coordinates")
    parser.add_argument("--public-showcase", action="store_true",
                        help="write the derived paired comparison to results/videos/human_to_panda.mp4")
    args = parser.parse_args()
    tracked_video = args.tracked_video.expanduser().resolve()
    trajectory_csv = args.trajectory_csv.expanduser().resolve()
    if not tracked_video.is_file():
        parser.error(f"tracked video not found: {tracked_video}")
    if not trajectory_csv.is_file():
        parser.error(f"trajectory CSV not found: {trajectory_csv}")
    if args.public_showcase and args.task_layout is None:
        parser.error("--public-showcase requires --task-layout for the contact-rich package replay")
    output_dir = args.output_dir.expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if output_dir == repo_root or repo_root in output_dir.parents:
        parser.error("output directory must be outside the repository to keep personal videos local")
    robot_output = output_dir / "panda_replay.mp4"
    paired_output = PUBLIC_SHOWCASE_PATH if args.public_showcase else output_dir / "human_panda_paired.mp4"
    try:
        _reject_output_input_aliases([robot_output, paired_output], [tracked_video, trajectory_csv])
    except ValueError as error:
        parser.error(str(error))
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.task_layout is None:
        points = read_trajectory(trajectory_csv)
        robot_video = render_panda_trajectory(points, robot_output, model_path=args.model_path)
    else:
        points = read_timed_trajectory(trajectory_csv)
        layout = TaskLayout.load(args.task_layout.expanduser())
        mapping = WorkspaceMapping(
            robot_x_min=args.robot_bounds[0], robot_x_max=args.robot_bounds[1],
            robot_y_min=args.robot_bounds[2], robot_y_max=args.robot_bounds[3],
        )
        pickup_xy, dropoff_xy = layout.to_robot_xy(mapping)
        robot_video = render_contact_panda_replay(
            points, pickup_xy, dropoff_xy, robot_output,
            human_label=args.human_label, model_path=args.model_path,
        )
    paired_video = make_paired_video(
        tracked_video,
        robot_video,
        paired_output,
        public_showcase=args.public_showcase,
    )
    print(f"Panda replay: {robot_video}")
    print(f"Paired demo: {paired_video}")


if __name__ == "__main__":
    main()
