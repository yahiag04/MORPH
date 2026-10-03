"""Render one Panda trajectory and pair it with a tracked real video locally."""

import argparse
import csv
from pathlib import Path

import numpy as np

from evaluation.demo_video import make_paired_video, render_panda_trajectory


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracked_video", type=Path, help="Annotated real demonstration MP4")
    parser.add_argument("trajectory_csv", type=Path, help="Retargeted XYZ CSV from the evaluator")
    parser.add_argument("--output-dir", required=True, type=Path, help="Local directory outside the repository")
    parser.add_argument("--model-path", type=Path, help="Optional Panda scene XML path")
    args = parser.parse_args()
    tracked_video = args.tracked_video.expanduser().resolve()
    trajectory_csv = args.trajectory_csv.expanduser().resolve()
    if not tracked_video.is_file():
        parser.error(f"tracked video not found: {tracked_video}")
    if not trajectory_csv.is_file():
        parser.error(f"trajectory CSV not found: {trajectory_csv}")
    output_dir = args.output_dir.expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if output_dir == repo_root or repo_root in output_dir.parents:
        parser.error("output directory must be outside the repository to keep personal videos local")
    output_dir.mkdir(parents=True, exist_ok=True)
    points = read_trajectory(trajectory_csv)
    robot_video = render_panda_trajectory(
        points,
        output_dir / "panda_replay.mp4",
        model_path=args.model_path,
    )
    paired_video = make_paired_video(
        tracked_video,
        robot_video,
        output_dir / "human_panda_paired.mp4",
    )
    print(f"Panda replay: {robot_video}")
    print(f"Paired demo: {paired_video}")


if __name__ == "__main__":
    main()
