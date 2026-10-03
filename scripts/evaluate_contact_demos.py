"""Run recorded pick-and-place paths through the MuJoCo contact task."""

from __future__ import annotations

import argparse
import json
import cv2
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.contact_demo_evaluator import aggregate_contact_runs, evaluate_contact_trajectory, write_transition_archive
from perception.retargeting import WorkspaceMapping, map_hand_trajectory
from perception.task_layout import TaskLayout, detect_task_layout
from perception.workspace_calibration import WorkspaceCalibration

METHODS = ("direct", "confidence-aware")


def _manifest_rows(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read processing manifest: {error}") from error
    if not isinstance(data, list):
        raise ValueError("processing manifest must be a list")
    return data


def _sample_control_rate(trajectory: np.ndarray, hz: float) -> np.ndarray:
    interval = 1.0 / hz
    indices = [0]
    last_time = float(trajectory[0, 0])
    for index in range(1, len(trajectory) - 1):
        if trajectory[index, 0] - last_time >= interval:
            indices.append(index)
            last_time = float(trajectory[index, 0])
    if len(trajectory) > 1 and indices[-1] != len(trajectory) - 1:
        indices.append(len(trajectory) - 1)
    sampled = trajectory[np.asarray(indices)]
    if len(sampled) > 1 and np.any(np.diff(sampled[:, 0]) <= 0):
        raise ValueError("downsampled trajectory timestamps are not strictly increasing")
    return sampled


def _plot(aggregates: dict[str, dict], destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    methods = list(aggregates)
    colors = ("#3572A5", "#E07A5F")
    labels = ("Human successful", "Human failed")
    label_keys = ("riusciti", "falliti")
    x = np.arange(len(methods))
    width = 0.34
    for offset, key, title in ((-width/2, "robot_success_rate", "Robot success rate"),):
        for index, (human_key, legend, color) in enumerate(zip(label_keys, labels, colors, strict=True)):
            values = [aggregates[method]["by_human_label"].get(human_key, {}).get(key) for method in methods]
            plotted = [0.0 if value is None else float(value) for value in values]
            axes[0].bar(x + offset + index * width, plotted, width, label=legend, color=color)
        axes[0].set_title(title)
        axes[0].set_ylabel("fraction of episodes")
        axes[0].set_ylim(0, 1)
        axes[0].set_xticks(x, methods, rotation=15)
        axes[0].legend(frameon=False)
        axes[0].grid(axis="y", alpha=.25)
    lifts = [aggregates[method]["mean_max_lift_m"] for method in methods]
    axes[1].bar(x, [0 if v is None else v for v in lifts], color=("#3572A5", "#E07A5F"))
    axes[1].set_title("Mean maximum package lift")
    axes[1].set_ylabel("meters")
    axes[1].set_xticks(x, methods, rotation=15)
    axes[1].grid(axis="y", alpha=.25)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processing-manifest", required=True, type=Path)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--task-layout", type=Path, help="local fallback layout JSON; object centers are detected per video when possible")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--robot-bounds", nargs=4, type=float, default=(0.38, 0.68, -0.22, 0.22),
                        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"))
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--simulation-steps-per-sample", type=int, default=50)
    parser.add_argument("--observation-steps", type=int, default=10,
                        help="physics steps between observations; must divide the controller interval")
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == repo_root or repo_root in output_dir.parents:
        parser.error("output directory must be outside the repository")
    if args.control_hz <= 0 or not np.isfinite(args.control_hz):
        parser.error("--control-hz must be finite and positive")

    manifest_path = args.processing_manifest.expanduser().resolve()
    calibration_dir = args.calibration_dir.expanduser().resolve()
    try:
        records = _manifest_rows(manifest_path)
        fallback_layout = TaskLayout.load(args.task_layout.expanduser()) if args.task_layout else None
        bounds = args.robot_bounds
        mapping = WorkspaceMapping(robot_x_min=bounds[0], robot_x_max=bounds[1],
                                    robot_y_min=bounds[2], robot_y_max=bounds[3])
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if not records:
        parser.error("processing manifest contains no clips")

    output_dir.mkdir(parents=True, exist_ok=True)
    results_by_method: dict[str, list[dict]] = {method: [] for method in METHODS}
    layout_counts = {"detected": 0, "fallback": 0, "unavailable": 0}
    for record in records:
        if not isinstance(record, dict) or not {"label", "source", "csv"}.issubset(record):
            for method in METHODS:
                results_by_method[method].append({"clip_id": "invalid_record", "human_label": "unknown",
                    "robot_success": False, "failure_reason": "malformed_manifest_record", "max_lift_m": 0.0, "steps": 0})
            continue
        label = str(record["label"])
        clip_id = Path(str(record["source"])).stem
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label) or not re.fullmatch(r"[A-Za-z0-9_-]+", clip_id):
            label, clip_id = "unknown", "invalid_clip"
        csv_path = Path(str(record["csv"])).expanduser()
        if not csv_path.is_absolute():
            csv_path = (manifest_path.parent / csv_path).resolve()
        clip_output = output_dir / "per_clip" / label / clip_id
        clip_output.mkdir(parents=True, exist_ok=True)
        try:
            calibration = WorkspaceCalibration.load(calibration_dir / label / f"{clip_id}.json")
            video_path = Path(str(record.get("annotated_video", ""))).expanduser()
            if not video_path.is_absolute():
                video_path = (manifest_path.parent / video_path).resolve()
            detected_layout = None
            if video_path.is_file():
                capture = cv2.VideoCapture(str(video_path))
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_count * 0.05)))
                ok, frame = capture.read()
                capture.release()
                if ok:
                    try:
                        detected_layout = detect_task_layout(frame, calibration.image_corners)
                    except ValueError:
                        detected_layout = None
            if detected_layout is not None:
                layout = detected_layout
                layout_source = "detected"
                layout_counts["detected"] += 1
            elif fallback_layout is not None:
                layout = fallback_layout
                layout_source = "fallback"
                layout_counts["fallback"] += 1
            else:
                raise ValueError("could not detect package and bowl; provide --task-layout as fallback")
            pickup, dropoff = layout.to_robot_xy(mapping)
            layout.save(clip_output / "task_layout.json")
            (clip_output / "layout_source.txt").write_text(layout_source + "\n", encoding="utf-8")
            preparation_error = None
        except (ValueError, OSError) as error:
            calibration = None
            preparation_error = str(error)
            layout_counts["unavailable"] += 1
        for method in METHODS:
            try:
                if preparation_error:
                    raise ValueError(preparation_error)
                mapped_path = clip_output / f"{method}_retargeted.csv"
                trajectory = map_hand_trajectory(
                    csv_path, mapped_path, mapping=mapping, confidence_threshold=args.confidence,
                    image_corners=calibration.image_corners, method=method,
                )
                trajectory = _sample_control_rate(trajectory, args.control_hz)
                run = evaluate_contact_trajectory(
                    trajectory, pickup, dropoff, human_label=label,
                    model_path=args.model_path, simulation_steps_per_sample=args.simulation_steps_per_sample,
                    observation_steps=args.observation_steps,
                )
                run["clip_id"] = clip_id
                run["method"] = method
                run["control_sample_count"] = int(len(trajectory))
                run["episode_id"] = f"{clip_id}:{method}"
                write_transition_archive([run], clip_output / f"{method}_transitions.npz")
                (clip_output / f"{method}_summary.json").write_text(json.dumps(
                    {k: v for k, v in run.items() if k not in {"final_package_xyz", "transitions"}}, indent=2) + "\n", encoding="utf-8")
            except (ValueError, OSError, RuntimeError) as error:
                run = {"clip_id": clip_id, "human_label": label, "method": method,
                       "robot_success": False, "failure_reason": str(error), "max_lift_m": 0.0,
                       "stable_in_tray": False, "ik_failures": 0, "steps": 0, "control_sample_count": 0}
                (clip_output / f"{method}_error.txt").write_text(str(error) + "\n", encoding="utf-8")
            results_by_method[method].append(run)
            print(f"{label}/{clip_id}/{method}: success={run['robot_success']} reason={run.get('failure_reason')}")

    aggregates = {method: aggregate_contact_runs(runs) for method, runs in results_by_method.items()}
    for method, runs in results_by_method.items():
        aggregate = aggregates[method]
        aggregate["retargeting_method"] = method
        (output_dir / f"{method}_aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
        write_transition_archive(runs, output_dir / f"{method}_transitions.npz")
    combined = {"dataset": {"clips": len(records), "human_label_counts": aggregates["direct"]["human_label_counts"]},
                "methods": aggregates, "task_layout_source_counts": layout_counts,
                "simulation_proxy": "MuJoCo rigid box and shallow tray; not physical robot validation"}
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    _plot(aggregates, output_dir / "aggregate_comparison.png")
    print(f"Aggregate results saved under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
