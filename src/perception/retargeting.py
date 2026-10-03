"""Map normalized camera coordinates to a fixed, calibrated Panda workspace."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .workspace_calibration import rectify_image_points


@dataclass(frozen=True)
class WorkspaceMapping:
    """Normalized camera rectangle and corresponding robot XY rectangle."""

    image_left: float = 0.2
    image_right: float = 0.8
    image_top: float = 0.15
    image_bottom: float = 0.85
    robot_x_min: float = 0.38
    robot_x_max: float = 0.68
    robot_y_min: float = -0.22
    robot_y_max: float = 0.22
    robot_z: float = 0.62

    def __post_init__(self):
        values = np.asarray((self.image_left, self.image_right, self.image_top, self.image_bottom, self.robot_x_min, self.robot_x_max, self.robot_y_min, self.robot_y_max, self.robot_z), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("workspace bounds and robot_z must all be finite")
        if not (0 <= self.image_left < self.image_right <= 1):
            raise ValueError("image horizontal bounds must satisfy 0 <= left < right <= 1")
        if not (0 <= self.image_top < self.image_bottom <= 1):
            raise ValueError("image vertical bounds must satisfy 0 <= top < bottom <= 1")
        if not self.robot_x_min < self.robot_x_max or not self.robot_y_min < self.robot_y_max:
            raise ValueError("robot workspace minimum bounds must be below maximum bounds")


def map_hand_trajectory(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    mapping: WorkspaceMapping = WorkspaceMapping(),
    confidence_threshold: float = 0.5,
    max_missing_fraction: float = 0.2,
    max_speed: float = 0.35,
    smoothing_window: int = 5,
    image_corners: np.ndarray | None = None,
    method: str = "direct",
) -> np.ndarray:
    """Interpolate gaps, optionally rectify XY, map it and save robot XYZ.

    ``direct`` preserves the original rectangular mapping and uniform smoothing.
    ``confidence-aware`` weights local smoothing by the tracker's score proxy.
    """
    if method not in {"direct", "confidence-aware"}:
        raise ValueError("method must be 'direct' or 'confidence-aware'")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between zero and one")
    if not 0 <= max_missing_fraction < 1:
        raise ValueError("max_missing_fraction must be in [0, 1)")
    if max_speed <= 0 or not np.isfinite(max_speed):
        raise ValueError("max_speed must be finite and positive")
    if smoothing_window < 1 or smoothing_window % 2 != 1:
        raise ValueError("smoothing_window must be a positive odd integer")

    with Path(input_csv).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("tracked CSV contains no frames")
    times = np.asarray([float(row["time"]) for row in rows])
    x = np.asarray([float(row["hand_x"]) if row["hand_x"] else np.nan for row in rows])
    y = np.asarray([float(row["hand_y"]) if row["hand_y"] else np.nan for row in rows])
    confidence = np.asarray([float(row["confidence"]) for row in rows])
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidence values must be finite and between zero and one")
    valid = np.isfinite(x) & np.isfinite(y) & (confidence >= confidence_threshold)
    if np.count_nonzero(valid) < 2:
        raise ValueError("at least two valid hand detections are required")
    if 1.0 - float(np.mean(valid)) > max_missing_fraction:
        raise ValueError("too many missing or low-confidence hand detections")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("frame timestamps must be strictly increasing")

    x = np.interp(times, times[valid], x[valid])
    y = np.interp(times, times[valid], y[valid])
    if image_corners is not None:
        normalized = rectify_image_points(np.column_stack((x, y)), image_corners)
        u = np.clip(normalized[:, 0], 0, 1)
        v = np.clip(normalized[:, 1], 0, 1)
    else:
        u = np.clip((x - mapping.image_left) / (mapping.image_right - mapping.image_left), 0, 1)
        v = np.clip((y - mapping.image_top) / (mapping.image_bottom - mapping.image_top), 0, 1)
    trajectory = np.column_stack((
        mapping.robot_x_min + u * (mapping.robot_x_max - mapping.robot_x_min),
        mapping.robot_y_min + v * (mapping.robot_y_max - mapping.robot_y_min),
        np.full(len(rows), mapping.robot_z),
    ))

    if method == "confidence-aware":
        trajectory = confidence_weighted_smooth(trajectory, confidence, smoothing_window)
    elif len(trajectory) > 2 and smoothing_window > 1:
        window = min(smoothing_window, len(trajectory) if len(trajectory) % 2 else len(trajectory) - 1)
        radius = window // 2
        padded = np.pad(trajectory, ((radius, radius), (0, 0)), mode="edge")
        trajectory = np.column_stack([np.convolve(padded[:, axis], np.ones(window) / window, mode="valid") for axis in range(3)])
    for index in range(1, len(trajectory)):
        distance = np.linalg.norm(trajectory[index] - trajectory[index - 1])
        allowed = max_speed * (times[index] - times[index - 1])
        if distance > allowed:
            trajectory[index] = trajectory[index - 1] + (trajectory[index] - trajectory[index - 1]) * (allowed / distance)
    if not np.isfinite(trajectory).all():
        raise ValueError("mapped trajectory contains non-finite coordinates")

    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("timestamp", "x", "y", "z"))
        for timestamp, point in zip(times, trajectory, strict=True):
            writer.writerow((f"{timestamp:.6f}", *(f"{coordinate:.7f}" for coordinate in point)))
    return np.column_stack((times, trajectory))


def confidence_weighted_smooth(
    points: np.ndarray,
    confidence: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Smooth XYZ points with a local mean weighted by tracker confidence.

    At a window with zero total weight, retain the interpolated center point.
    This keeps missing/invalid observations finite without treating them as
    trusted measurements.
    """
    try:
        trajectory = np.asarray(points, dtype=np.float64)
        weights = np.asarray(confidence, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("points and confidence must be numeric arrays") from error
    if trajectory.ndim != 2 or trajectory.shape[1] != 3 or trajectory.shape[0] < 1:
        raise ValueError("points must have shape (N, 3) with at least one row")
    if not np.isfinite(trajectory).all():
        raise ValueError("points must contain only finite coordinates")
    if weights.shape != (trajectory.shape[0],):
        raise ValueError("confidence must have one value per point")
    if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError("confidence values must be finite and between zero and one")
    if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)) or window_size < 1 or window_size % 2 != 1:
        raise ValueError("window_size must be a positive odd integer")
    largest_odd = trajectory.shape[0] if trajectory.shape[0] % 2 else trajectory.shape[0] - 1
    window = min(int(window_size), max(1, largest_odd))
    if window == 1:
        return trajectory.copy()

    radius = window // 2
    padded_points = np.pad(trajectory, ((radius, radius), (0, 0)), mode="edge")
    padded_weights = np.pad(weights, (radius, radius), mode="edge")
    smoothed = np.empty_like(trajectory)
    for index in range(len(trajectory)):
        local_points = padded_points[index : index + window]
        local_weights = padded_weights[index : index + window]
        total_weight = float(local_weights.sum())
        if total_weight > 0:
            smoothed[index] = np.average(local_points, axis=0, weights=local_weights)
        else:
            smoothed[index] = trajectory[index]
    return smoothed
