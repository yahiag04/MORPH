"""Tests for marker-based perspective calibration."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from perception.workspace_calibration import (
    WorkspaceCalibration,
    detect_workspace_corners,
    rectify_image_points,
)


def pink_marker_frame(points, size=(640, 480)):
    width, height = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    color = cv2.cvtColor(np.uint8([[[160, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    for x, y in points:
        center = (round(x * width), round(y * height))
        cv2.rectangle(frame, (center[0] - 18, center[1] - 18), (center[0] + 18, center[1] + 18), tuple(int(c) for c in color), -1)
    return frame


class WorkspaceCalibrationTests(unittest.TestCase):
    def test_detects_corners_in_clockwise_image_order(self):
        expected = np.asarray(((0.15, 0.15), (0.82, 0.2), (0.78, 0.84), (0.12, 0.88)))
        actual = detect_workspace_corners(pink_marker_frame(expected))
        np.testing.assert_allclose(actual, expected, atol=0.003)

    def test_rectifies_known_projective_corners_to_unit_square(self):
        corners = np.asarray(((0.15, 0.15), (0.82, 0.2), (0.78, 0.84), (0.12, 0.88)))
        actual = rectify_image_points(corners, corners)
        np.testing.assert_allclose(actual, ((0, 0), (1, 0), (1, 1), (0, 1)), atol=1e-6)

    def test_rejects_missing_marker(self):
        points = ((0.15, 0.15), (0.82, 0.2), (0.78, 0.84))
        with self.assertRaisesRegex(ValueError, "four pink workspace markers"):
            detect_workspace_corners(pink_marker_frame(points))

    def test_ignores_internal_pink_object_inside_workspace_corners(self):
        points = ((0.15, 0.15), (0.82, 0.2), (0.78, 0.84), (0.12, 0.88), (0.5, 0.5))
        try:
            actual = detect_workspace_corners(pink_marker_frame(points))
        except ValueError as error:
            self.fail(f"internal pink object should not make the corners ambiguous: {error}")
        np.testing.assert_allclose(actual, np.asarray(points[:4]), atol=0.003)

    def test_rejects_extra_pink_candidate_on_outer_hull(self):
        points = ((0.15, 0.15), (0.82, 0.2), (0.78, 0.84), (0.12, 0.88), (0.97, 0.5))
        with self.assertRaisesRegex(ValueError, "four pink workspace markers"):
            detect_workspace_corners(pink_marker_frame(points))

    def test_calibration_json_round_trip(self):
        expected = np.asarray(((0.15, 0.15), (0.82, 0.2), (0.78, 0.84), (0.12, 0.88)))
        calibration = WorkspaceCalibration(expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            calibration.save(path)
            restored = WorkspaceCalibration.load(path)
            np.testing.assert_allclose(restored.image_corners, expected)
            self.assertEqual(json.loads(path.read_text())["version"], 1)


if __name__ == "__main__":
    unittest.main()
