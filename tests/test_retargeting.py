"""Tests for normalized image-to-workspace mapping and trajectory guards."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from perception.retargeting import WorkspaceMapping, map_hand_trajectory


def write_tracked(path: Path, points):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "time", "hand_x", "hand_y", "hand_z", "confidence"))
        for index, (x, y, confidence) in enumerate(points):
            writer.writerow((index, index / 10, x, y, "", confidence))


class RetargetingTests(unittest.TestCase):
    def test_rejects_nonfinite_mapping_values_and_timestamps(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            WorkspaceMapping(robot_z=float("nan"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tracked.csv"
            write_tracked(source, [(0.2, 0.5, 0.9), (0.8, 0.5, 0.9)])
            content = source.read_text(encoding="utf-8").replace("0.1", "nan")
            source.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "timestamps"):
                map_hand_trajectory(source, root / "robot.csv")

    def test_maps_image_rectangle_corners_to_robot_bounds_and_fixed_z(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "tracked.csv", root / "robot.csv"
            write_tracked(source, [(0.2, 0.15, 0.9), (0.8, 0.85, 0.9)])
            trajectory = map_hand_trajectory(source, output, smoothing_window=1, max_speed=10)
            np.testing.assert_allclose(trajectory[0, 1:], [0.38, 0.22, 0.62])
            np.testing.assert_allclose(trajectory[1, 1:], [0.68, -0.22, 0.62])
            self.assertTrue(output.is_file())

    def test_interpolates_single_missing_sample_and_caps_velocity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tracked.csv"
            write_tracked(source, [(0.2, 0.5, 0.9), ("", "", 0), (0.8, 0.5, 0.9)])
            result = map_hand_trajectory(source, root / "robot.csv", max_speed=0.1, smoothing_window=1, max_missing_fraction=0.34)
            self.assertAlmostEqual(result[1, 1], 0.39, places=5)
            speeds = np.linalg.norm(np.diff(result[:, 1:], axis=0), axis=1) / 0.1
            self.assertTrue(np.all(speeds <= 0.100001))

    def test_rejects_excessive_missing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tracked.csv"
            write_tracked(source, [(0.2, 0.5, 0.9), ("", "", 0), ("", "", 0), (0.8, 0.5, 0.9)])
            with self.assertRaisesRegex(ValueError, "too many"):
                map_hand_trajectory(source, Path(directory) / "robot.csv")


if __name__ == "__main__":
    unittest.main()
