"""Detect pink workspace markers and rectify hand points into a unit square."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _validated_corners(corners: np.ndarray) -> np.ndarray:
    """Validate normalized TL, TR, BR, BL image corners and return a copy."""
    try:
        points = np.asarray(corners, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("image_corners must contain numeric coordinates") from error
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("image_corners must be a finite array with shape (4, 2)")
    if np.any(points < 0) or np.any(points > 1):
        raise ValueError("image_corners must be normalized between zero and one")

    edges = np.roll(points, -1, axis=0) - points
    next_edges = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    if np.any(np.abs(cross) < 1e-8) or not (np.all(cross > 0) or np.all(cross < 0)):
        raise ValueError("image_corners must form a non-degenerate convex quadrilateral")
    area = 0.5 * abs(
        np.dot(points[:, 0], np.roll(points[:, 1], -1))
        - np.dot(points[:, 1], np.roll(points[:, 0], -1))
    )
    if area < 1e-4:
        raise ValueError("image_corners quadrilateral is too small to calibrate")
    return points.copy()


@dataclass(frozen=True, init=False)
class WorkspaceCalibration:
    """Normalized image centers for workspace corners ordered TL, TR, BR, BL."""

    _corners: tuple[tuple[float, float], ...]

    def __init__(self, image_corners: np.ndarray):
        points = _validated_corners(image_corners)
        object.__setattr__(
            self, "_corners", tuple(tuple(float(value) for value in point) for point in points)
        )

    @property
    def image_corners(self) -> np.ndarray:
        """Return a copy of the four normalized image corners."""
        return np.asarray(self._corners, dtype=np.float64)

    def save(self, path: str | Path) -> Path:
        """Write a versioned JSON calibration file."""
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "image_corners": self.image_corners.tolist()}
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "WorkspaceCalibration":
        """Load and validate a version 1 calibration JSON file."""
        source = Path(path).expanduser()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read workspace calibration '{source}': {error}") from error
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError(f"Unsupported workspace calibration format in '{source}'")
        if "image_corners" not in payload:
            raise ValueError(f"Calibration '{source}' is missing image_corners")
        return cls(np.asarray(payload["image_corners"], dtype=np.float64))


def detect_workspace_corners(frame: np.ndarray) -> np.ndarray:
    """Detect exactly four pink marker centers in normalized TL, TR, BR, BL order.

    The HSV interval covers saturated pink/magenta markers. Candidate contours
    must be compact and large enough relative to the frame to avoid counting
    isolated color noise as a workspace marker.
    """
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("frame must be a non-empty BGR image with three channels")
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((140, 70, 100)), np.array((179, 255, 255)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = max(100.0, height * width * 0.00015)
    candidates: list[tuple[float, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        ratio = box_width / box_height if box_height else 0.0
        if not 0.55 <= ratio <= 1.8:
            continue
        if box_width / width > 0.08 or box_height / height > 0.08:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        center_x = moments["m10"] / moments["m00"] / width
        center_y = moments["m01"] / moments["m00"] / height
        candidates.append((center_x, center_y))

    if len(candidates) < 4:
        raise ValueError(
            f"Expected four pink workspace markers, found {len(candidates)} usable candidates"
        )

    points = np.asarray(candidates, dtype=np.float64)
    hull_indices = cv2.convexHull(
        points.astype(np.float32).reshape(-1, 1, 2), returnPoints=False
    ).reshape(-1)
    if hull_indices.size != 4:
        raise ValueError(
            "Expected four pink workspace markers on the outer boundary, "
            f"found {hull_indices.size} outer candidates from {len(candidates)} usable candidates"
        )
    points = points[hull_indices]
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    clockwise = points[np.argsort(angles)]
    start = int(np.argmin(clockwise.sum(axis=1)))
    ordered = np.roll(clockwise, -start, axis=0)
    return _validated_corners(ordered)


def rectify_image_points(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Project normalized image points through a corner calibration to unit XY."""
    try:
        coordinates = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("points must contain numeric normalized XY coordinates") from error
    was_single = coordinates.shape == (2,)
    if was_single:
        coordinates = coordinates.reshape(1, 2)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError("points must be a finite array with shape (N, 2)")
    source_corners = _validated_corners(corners).astype(np.float32)
    destination_corners = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source_corners, destination_corners)
    if not np.isfinite(transform).all() or abs(float(np.linalg.det(transform))) < 1e-12:
        raise ValueError("workspace corners do not define a stable perspective transform")
    projected = cv2.perspectiveTransform(coordinates.astype(np.float32).reshape(-1, 1, 2), transform)
    result = projected.reshape(-1, 2).astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError("rectified points contain non-finite coordinates")
    return result[0] if was_single else result
