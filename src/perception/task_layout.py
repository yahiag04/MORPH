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
