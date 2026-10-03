"""Run direct and confidence-aware Panda replay on a local demo manifest."""

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.demo_evaluator import evaluate_trajectory
from evaluation.demo_metrics import tracking_metrics, trajectory_metrics
from perception.retargeting import WorkspaceMapping, map_hand_trajectory
from perception.workspace_calibration import WorkspaceCalibration


METHODS = ("direct", "confidence-aware")
METRIC_NAMES = (
    "coverage_fraction",
    "interpolated_frames",
    "path_length_m",
    "jerk_rms_mps3",
    "ik_nonconverged_count",
    "mean_position_error_m",
    "max_position_error_m",
)


def _manifest_rows(path: Path) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read processing manifest '{path}': {error}") from error
    if not isinstance(payload, list) or not payload:
        raise ValueError("processing manifest must be a non-empty list of clip records")
    return payload


def _read_tracking_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"tracked CSV contains no frames: {path}")
    times = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
    confidence = np.asarray([float(row["confidence"]) for row in rows], dtype=np.float64)
    return times, confidence


def _aggregate(runs: dict[str, list[dict]], labels: dict[str, int], failures: int) -> dict:
    methods: dict[str, dict] = {}
    for method in METHODS:
        rows = runs[method]
        metrics = {}
        for name in METRIC_NAMES:
            values = np.asarray(
                [row[name] for row in rows if row.get(name) is not None], dtype=np.float64
            )
            if values.size:
                metrics[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
            else:
                metrics[name] = {"mean": None, "std": None}
        methods[method] = {"completed_runs": len(rows), "metrics": metrics}
    return {
        "dataset": {"clips": int(sum(labels.values())), "human_label_counts": labels},
        "methods": methods,
        "failed_runs": int(failures),
    }


def _plot_aggregate(aggregate: dict, output_path: Path) -> None:
    metrics = (
        ("coverage_fraction", "Tracking coverage", "fraction"),
        ("jerk_rms_mps3", "RMS Cartesian jerk", "m/s³"),
        ("mean_position_error_m", "Mean Panda position error", "m"),
        ("ik_nonconverged_count", "IK non-converged samples", "per clip"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, (key, title, unit) in zip(axes.flat, metrics, strict=True):
        positions = np.arange(len(METHODS))
        available = []
        for method, position, color in zip(METHODS, positions, ("#247ba0", "#70c1b3"), strict=True):
            value = aggregate["methods"][method]["metrics"][key]
            mean = value["mean"]
            if mean is None:
                continue
            deviation = value["std"] if value["std"] is not None else 0.0
            axis.bar(position, mean, yerr=deviation, capsize=4, color=color)
            available.append(float(mean) + float(deviation))
        axis.set_xticks(positions, ("Direct", "Confidence-aware"), rotation=10)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.25)
        if available:
            axis.set_ylim(bottom=0, top=max(available) * 1.18 or 1.0)
            missing_y = axis.get_ylim()[1] * 0.04
        else:
            axis.set_ylim(0, 1)
            missing_y = 0.05
        for method, position in zip(METHODS, positions, strict=True):
            if aggregate["methods"][method]["metrics"][key]["mean"] is None:
                axis.text(position, missing_y, "N/A", ha="center", va="bottom", color="#555555")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processing-manifest", required=True, type=Path)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--robot-bounds", nargs=4, type=float, default=(0.38, 0.68, -0.22, 0.22), metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"))
    parser.add_argument("--robot-z", type=float, default=0.62)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--max-missing-fraction", type=float, default=0.2)
    parser.add_argument("--max-speed", type=float, default=0.35)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == repo_root or repo_root in output_dir.parents:
        parser.error("output directory must be outside the repository to keep personal trajectories local")
    manifest_path = args.processing_manifest.expanduser().resolve()
    calibration_dir = args.calibration_dir.expanduser().resolve()
    try:
        records = _manifest_rows(manifest_path)
    except ValueError as error:
        parser.error(str(error))

    bounds = args.robot_bounds
    mapping = WorkspaceMapping(
        robot_x_min=bounds[0], robot_x_max=bounds[1],
        robot_y_min=bounds[2], robot_y_max=bounds[3], robot_z=args.robot_z,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {method: [] for method in METHODS}
    label_counts: dict[str, int] = {}
    failures = 0

    for record in records:
        if not isinstance(record, dict) or not {"label", "source", "csv"}.issubset(record):
            print(f"FAILED malformed manifest record: {record!r}")
            failures += len(METHODS)
            continue
        label = str(record["label"])
        source_name = Path(str(record["source"])).name
        clip_id = Path(source_name).stem
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label) or not re.fullmatch(r"[A-Za-z0-9_-]+", clip_id):
            print(f"FAILED invalid label or clip id: {label}/{clip_id}")
            failures += len(METHODS)
            continue
        label_counts[label] = label_counts.get(label, 0) + 1
        tracked_csv = Path(record["csv"]).expanduser()
        if not tracked_csv.is_absolute():
            tracked_csv = (manifest_path.parent / tracked_csv).resolve()
        calibration_path = calibration_dir / label / f"{clip_id}.json"
        local_clip_dir = output_dir / "per_clip" / label / clip_id
        local_clip_dir.mkdir(parents=True, exist_ok=True)

        try:
            calibration = WorkspaceCalibration.load(calibration_path)
            times, confidence = _read_tracking_csv(tracked_csv)
            tracking = tracking_metrics(times, confidence, args.confidence)
        except (OSError, ValueError, KeyError) as error:
            for method in METHODS:
                failures += 1
                error_path = local_clip_dir / f"{method}_error.txt"
                error_path.write_text(str(error) + "\n", encoding="utf-8")
                print(f"FAILED {label}/{clip_id}/{method}: {error}")
            continue

        for method in METHODS:
            try:
                trajectory_path = local_clip_dir / f"{method}_trajectory.csv"
                mapped = map_hand_trajectory(
                    tracked_csv,
                    trajectory_path,
                    mapping=mapping,
                    confidence_threshold=args.confidence,
                    max_missing_fraction=args.max_missing_fraction,
                    max_speed=args.max_speed,
                    image_corners=calibration.image_corners,
                    method=method,
                )
                path_metrics = trajectory_metrics(mapped[:, 0], mapped[:, 1:])
                log, simulation_metrics = evaluate_trajectory(mapped[:, 1:], args.model_path)
                log_path = local_clip_dir / f"{method}_simulation.csv"
                log.write_csv(log_path)
                summary = {
                    "clip_id": clip_id,
                    "human_label": label,
                    "method": method,
                    "tracking_sample_count": tracking["sample_count"],
                    "detected_frames": tracking["detected_frames"],
                    "interpolated_frames": tracking["interpolated_frames"],
                    "coverage_fraction": tracking["coverage_fraction"],
                    **path_metrics,
                    **{f"simulation_{key}": value for key, value in simulation_metrics.items()},
                    "mean_position_error_m": simulation_metrics["mean_position_error_m"],
                    "max_position_error_m": simulation_metrics["max_position_error_m"],
                    "ik_nonconverged_count": simulation_metrics["ik_nonconverged_count"],
                }
                (local_clip_dir / f"{method}_summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
                )
                runs[method].append(summary)
                print(
                    f"OK {label}/{clip_id}/{method}: "
                    f"{simulation_metrics['sample_count']} sim samples, "
                    f"mean error {simulation_metrics['mean_position_error_m']:.4f} m"
                )
            except (OSError, ValueError, RuntimeError) as error:
                failures += 1
                (local_clip_dir / f"{method}_error.txt").write_text(
                    str(error) + "\n", encoding="utf-8"
                )
                print(f"FAILED {label}/{clip_id}/{method}: {error}")

    aggregate = _aggregate(runs, label_counts, failures)
    aggregate_path = output_dir / "aggregate_metrics.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    figure_path = output_dir / "method_comparison.png"
    _plot_aggregate(aggregate, figure_path)
    completed = sum(item["completed_runs"] for item in aggregate["methods"].values())
    print(f"Completed {completed}/{len(records) * len(METHODS)} runs; failures: {failures}")
    print(f"Aggregate metrics: {aggregate_path}")
    print(f"Aggregate figure: {figure_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
