"""Cartesian waypoint interpolation and smoothing utilities."""

from __future__ import annotations

import numpy as np


def _positive_finite_scalar(name: str, value: float) -> float:
    """Validate and normalize a positive finite numeric scalar."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite positive number")
    if not np.isrealobj(value):
        raise ValueError(f"{name} must be a finite positive real number")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric_value


def _waypoint_array(waypoints: np.ndarray) -> np.ndarray:
    """Return validated, copied XYZ waypoints with repeated segments removed."""
    try:
        points = np.array(waypoints, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("waypoints must contain numeric XYZ coordinates") from error
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("waypoints must have shape (N, 3)")
    if points.shape[0] < 2:
        raise ValueError("waypoints must contain at least two points")
    if not np.isfinite(points).all():
        raise ValueError("waypoint coordinates must all be finite")

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 0.0))
    return points[keep]


def resample_waypoints(
    waypoints: np.ndarray,
    speed: float,
    sample_period: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample an XYZ path at constant arc-length speed.

    Consecutive duplicate points are ignored. The returned timestamps start at
    zero and include the path endpoint, even when its time is not a multiple of
    ``sample_period``.

    Args:
        waypoints: Finite array with shape ``(N, 3)`` and at least two points.
        speed: Positive Cartesian path speed in meters per second.
        sample_period: Positive time between regular samples in seconds.

    Returns:
        A pair ``(timestamps, xyz_samples)`` with the final endpoint included.

    Raises:
        ValueError: If the path, speed, or sample period is invalid.
    """
    points = _waypoint_array(waypoints)
    speed = _positive_finite_scalar("speed", speed)
    sample_period = _positive_finite_scalar("sample_period", sample_period)

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_distance = float(segment_lengths.sum())
    if not np.isfinite(total_distance) or total_distance <= 0.0:
        raise ValueError("waypoints must define a nonzero path distance")

    duration = total_distance / speed
    timestamps = np.arange(0.0, duration, sample_period, dtype=np.float64)
    if timestamps.size == 0:
        timestamps = np.array([0.0], dtype=np.float64)
    if duration - timestamps[-1] > 1e-12:
        timestamps = np.append(timestamps, duration)
    else:
        timestamps[-1] = duration

    distances = np.minimum(speed * timestamps, total_distance)
    cumulative_distance = np.concatenate(
        ([0.0], np.cumsum(segment_lengths, dtype=np.float64))
    )
    xyz_samples = np.column_stack(
        [np.interp(distances, cumulative_distance, points[:, axis]) for axis in range(3)]
    )
    xyz_samples[0] = points[0]
    xyz_samples[-1] = points[-1]
    return timestamps, xyz_samples


def smooth_trajectory(samples: np.ndarray, window_size: int) -> np.ndarray:
    """Smooth interior XYZ samples with a moving average and fixed endpoints.

    When ``window_size`` exceeds the available samples, the effective window
    is capped to the largest usable odd size. A size of one returns a copy.
    """
    try:
        points = np.array(samples, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("samples must contain numeric XYZ coordinates") from error
    if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] < 1:
        raise ValueError("samples must have shape (N, 3) with at least one row")
    if not np.isfinite(points).all():
        raise ValueError("sample coordinates must all be finite")
    if (
        isinstance(window_size, bool)
        or not isinstance(window_size, (int, np.integer))
        or window_size < 1
        or window_size % 2 == 0
    ):
        raise ValueError("window_size must be a positive odd integer")

    largest_odd_window = points.shape[0] if points.shape[0] % 2 else points.shape[0] - 1
    effective_window = min(int(window_size), max(1, largest_odd_window))
    if effective_window == 1 or points.shape[0] < 3:
        return points

    radius = effective_window // 2
    padded = np.pad(points, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.full(effective_window, 1.0 / effective_window)
    smoothed = np.column_stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(3)]
    )
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed
