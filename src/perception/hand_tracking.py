"""Small helpers for turning detected hand landmarks into palm observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PALM_LANDMARK_INDICES = (0, 5, 9, 13, 17)
MAX_PALM_STEP_NORMALIZED = 0.25


@dataclass(frozen=True)
class PalmObservation:
    x: float
    y: float
    confidence: float


def palm_observation(
    result: Any, previous_xy: tuple[float, float] | None = None
) -> PalmObservation | None:
    """Select the most confident hand and average its five palm landmarks."""
    landmarks = getattr(result, "hand_landmarks", None) or []
    handedness = getattr(result, "handedness", None) or []
    candidates = []
    for index, hand in enumerate(landmarks):
        score = 1.0
        if index < len(handedness) and handedness[index]:
            score = float(handedness[index][0].score)
        if len(hand) > max(PALM_LANDMARK_INDICES):
            points = [hand[landmark_index] for landmark_index in PALM_LANDMARK_INDICES]
            x = sum(float(point.x) for point in points) / len(points)
            y = sum(float(point.y) for point in points) / len(points)
            candidates.append((score, hand, x, y))
    if not candidates:
        return None
    if previous_xy is None:
        confidence, _, x, y = max(candidates, key=lambda candidate: candidate[0])
    else:
        candidate = min(
            candidates,
            key=lambda candidate: (candidate[2] - previous_xy[0]) ** 2
            + (candidate[3] - previous_xy[1]) ** 2,
        )
        confidence, _, x, y = candidate
        if (x - previous_xy[0]) ** 2 + (y - previous_xy[1]) ** 2 > MAX_PALM_STEP_NORMALIZED**2:
            return None
    return PalmObservation(
        x=x,
        y=y,
        confidence=confidence,
    )
