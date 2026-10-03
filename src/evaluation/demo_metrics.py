"""Metrics for hand-tracking coverage and Cartesian trajectory quality."""

from __future__ import annotations

import numpy as np


def tracking_metrics(
    times: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Report detected coverage and frames that retargeting must interpolate."""
    timestamps = np.asarray(times, dtype=np.float64)
    scores = np.asarray(confidence, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size < 1 or not np.isfinite(timestamps).all():
        raise ValueError("times must contain at least one finite timestamp")
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if scores.shape != timestamps.shape or not np.isfinite(scores).all():
        raise ValueError("confidence must contain one finite score per timestamp")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("confidence values must be between zero and one")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between zero and one")

    detected_frames = int(np.count_nonzero(scores >= threshold))
    sample_count = int(scores.size)
    return {
        "sample_count": sample_count,
        "detected_frames": detected_frames,
        "interpolated_frames": sample_count - detected_frames,
        "coverage_fraction": detected_frames / sample_count,
    }


def trajectory_metrics(times: np.ndarray, xyz: np.ndarray) -> dict[str, float | None]:
    """Return path length in meters and RMS Cartesian jerk in m/s³.

    Three timestamp-aware gradients are used for jerk. Jerk is zero for a
    stationary trajectory and undefined (``None``) when fewer than four
    samples are available.
    """
    timestamps = np.asarray(times, dtype=np.float64)
    points = np.asarray(xyz, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size < 1:
        raise ValueError("times must contain at least one timestamp")
    if points.shape != (timestamps.size, 3) or not np.isfinite(points).all():
        raise ValueError("xyz must contain one finite three-dimensional point per timestamp")
    if not np.isfinite(timestamps).all():
        raise ValueError("timestamps must be finite")
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")

    if timestamps.size == 1:
        path_length = 0.0
    else:
        path_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if path_length <= 1e-12:
        jerk_rms: float | None = 0.0
    elif timestamps.size < 4:
        jerk_rms = None
    else:
        velocity = np.gradient(points, timestamps, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, timestamps, axis=0, edge_order=2)
        jerk = np.gradient(acceleration, timestamps, axis=0, edge_order=2)
        if not np.isfinite(jerk).all():
            raise ValueError("computed trajectory jerk contains non-finite values")
        jerk_rms = float(np.sqrt(np.mean(np.sum(jerk * jerk, axis=1))))
    return {"sample_count": int(timestamps.size), "path_length_m": path_length, "jerk_rms_mps3": jerk_rms}
