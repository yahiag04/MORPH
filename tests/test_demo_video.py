"""Tests for local real/Panda side-by-side video assembly."""

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from evaluation.demo_video import make_paired_video


def write_color_video(path: Path, colors, size=(96, 64), fps=10.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("Could not create test video")
    for color in colors:
        frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def read_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    return frames


class DemoVideoTests(unittest.TestCase):
    def test_pairs_sources_with_normalized_progress_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real.mp4"
            robot = root / "robot.mp4"
            output = root / "paired.mp4"
            write_color_video(real, [(20, 20, 20), (80, 80, 80)], size=(64, 96))
            write_color_video(robot, [(30, 30, 30), (60, 60, 60), (120, 120, 120), (200, 200, 200)])
            real_bytes, robot_bytes = real.read_bytes(), robot.read_bytes()

            result = make_paired_video(real, robot, output, fps=10.0)
            frames = read_frames(result)

            self.assertEqual(result, output.resolve())
            self.assertEqual(len(frames), 4)
            self.assertEqual(frames[0].shape, (688, 1000, 3))
            self.assertGreater(np.count_nonzero(frames[0][:48] > 180), 0)
            # The shorter source holds each endpoint across normalized progress.
            human_panel = frames[1][48 + 320, 180]
            self.assertLess(abs(int(human_panel[0]) - 20), 12)
            human_panel_end = frames[2][48 + 320, 180]
            self.assertLess(abs(int(human_panel_end[0]) - 80), 12)
            self.assertEqual(real.read_bytes(), real_bytes)
            self.assertEqual(robot.read_bytes(), robot_bytes)

    def test_rejects_output_that_would_overwrite_an_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real.mp4"
            robot = root / "robot.mp4"
            write_color_video(real, [(20, 20, 20)])
            write_color_video(robot, [(80, 80, 80)])
            original = real.read_bytes()
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                make_paired_video(real, robot, real)
            self.assertEqual(real.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
