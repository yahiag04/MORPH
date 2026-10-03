"""Local, normalized pickup and dropoff points for overhead demonstrations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .retargeting import WorkspaceMapping


@dataclass(frozen=True)
class TaskLayout:
    """Pickup and dropoff points in a rectified workspace unit square."""

    pickup_uv: tuple[float, float]
    dropoff_uv: tuple[float, float]

    def __post_init__(self) -> None:
        for name in ("pickup_uv", "dropoff_uv"):
            try:
                point = np.asarray(getattr(self, name), dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name} must be a finite point in the unit square") from error
            if point.shape != (2,) or not np.isfinite(point).all() or np.any(point < 0) or np.any(point > 1):
                raise ValueError(f"{name} must be a finite point in the unit square")
            object.__setattr__(self, name, (float(point[0]), float(point[1])))

    @classmethod
    def load(cls, path: str | Path) -> "TaskLayout":
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("task layout JSON must be an object")
        try:
            return cls(tuple(payload["pickup_uv"]), tuple(payload["dropoff_uv"]))
        except (KeyError, TypeError) as error:
            raise ValueError("task layout must contain pickup_uv and dropoff_uv") from error

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"pickup_uv": self.pickup_uv, "dropoff_uv": self.dropoff_uv}, indent=2) + "\n", encoding="utf-8")

    def to_robot_xy(self, mapping: WorkspaceMapping) -> tuple[np.ndarray, np.ndarray]:
        """Map normalized UV points into the calibrated robot XY rectangle."""
        def convert(uv: tuple[float, float]) -> np.ndarray:
            u, v = uv
            return np.array((mapping.robot_x_min + u * (mapping.robot_x_max - mapping.robot_x_min),
                             mapping.robot_y_min + v * (mapping.robot_y_max - mapping.robot_y_min)), dtype=np.float64)
        return convert(self.pickup_uv), convert(self.dropoff_uv)


def detect_task_layout(frame: np.ndarray, image_corners: np.ndarray) -> TaskLayout:
    """Estimate package and bowl UV centers in one calibrated video frame.

    The detector uses the navy package wrapper and wood-toned receiving bowl
    present in the current recording setup. Keep the resulting coordinates
    local to the user's dataset.
    """
    import cv2

    from .workspace_calibration import rectify_image_points

    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("frame must be a non-empty BGR image")
    height, width = image.shape[:2]
    corners = np.asarray(image_corners, dtype=np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("image_corners must have shape (4, 2) and finite values")
    polygon = np.round(corners * np.asarray((width, height))).astype(np.int32)
    roi = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(roi, [polygon], 255)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def components(mask: np.ndarray, *, close: int = 1) -> list[tuple[float, np.ndarray, tuple[int, int, int, int]]]:
        mask = cv2.bitwise_and(mask, roi)
        if close > 1:
            kernel = np.ones((close, close), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
        return [(float(stats[index, cv2.CC_STAT_AREA]) / (width * height),
                 centers[index] / np.asarray((width, height)),
                 tuple(int(value) for value in stats[index, :4]))
                for index in range(1, count)]

    brown = cv2.inRange(hsv, np.asarray((5, 45, 25)), np.asarray((30, 255, 220)))
    bowl_candidates = []
    for area, center, (x, y, box_width, box_height) in components(brown, close=7):
        aspect = box_width / max(1, box_height)
        if (0.012 <= area <= 0.07 and 0.10 <= box_width / width <= 0.40
                and 0.06 <= box_height / height <= 0.30 and 0.60 <= aspect <= 1.40):
            bowl_candidates.append((area, center))
    if not bowl_candidates:
        raise ValueError("could not detect the wooden bowl inside the calibrated workspace")
    bowl_uv = max(bowl_candidates, key=lambda candidate: candidate[0])[1]

    blue = cv2.inRange(hsv, np.asarray((85, 20, 5)), np.asarray((135, 255, 190)))
    package_candidates = []
    for area, center, (x, y, box_width, box_height) in components(blue, close=19):
        aspect = box_width / max(1, box_height)
        if (0.002 <= area <= 0.04 and 0.035 <= box_width / width <= 0.24
                and 0.025 <= box_height / height <= 0.22 and 0.25 <= aspect <= 2.8):
            package_candidates.append((area, center))
    if not package_candidates:
        dark = cv2.inRange(hsv, np.asarray((0, 0, 0)), np.asarray((179, 255, 75)))
        for area, center, (x, y, box_width, box_height) in components(dark, close=25):
            aspect = box_width / max(1, box_height)
            if (0.0015 <= area <= 0.03 and 0.035 <= box_width / width <= 0.24
                    and 0.025 <= box_height / height <= 0.22 and 0.25 <= aspect <= 2.8):
                package_candidates.append((area, center))
    if not package_candidates:
        raise ValueError("could not detect the navy package inside the calibrated workspace")
    package_xy = max(package_candidates, key=lambda candidate: candidate[0])[1]
    pickup_uv = rectify_image_points(package_xy, corners)
    dropoff_uv = rectify_image_points(bowl_uv, corners)
    return TaskLayout(tuple(pickup_uv), tuple(dropoff_uv))
