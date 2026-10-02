"""Tests for frame-aligned hand tracking output using detector fixtures."""

import csv
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from perception.hand_tracking import palm_observation
from perception.video_processor import ensure_hand_landmarker_model, process_video


def make_video(path: Path, frame_count: int = 4) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("Could not create test video")
    for _ in range(frame_count):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()


def fake_result(x: float, y: float):
    points = [SimpleNamespace(x=x, y=y) for _ in range(21)]
    return SimpleNamespace(hand_landmarks=[points], handedness=[[SimpleNamespace(score=0.87)]])


class FakeLandmarker:
    def __init__(self):
        self.timestamps = []

    def detect_for_video(self, image, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        return fake_result(0.25, 0.75) if len(self.timestamps) != 2 else SimpleNamespace(hand_landmarks=[], handedness=[])


class VideoProcessorTests(unittest.TestCase):
    def test_palm_observation_averages_five_landmarks_and_reads_confidence(self):
        hand = [SimpleNamespace(x=index / 20, y=index / 40) for index in range(21)]
        result = palm_observation(SimpleNamespace(hand_landmarks=[hand], handedness=[[SimpleNamespace(score=0.8)]]))
        self.assertAlmostEqual(result.x, sum(i / 20 for i in (0, 5, 9, 13, 17)) / 5)
        self.assertAlmostEqual(result.y, sum(i / 40 for i in (0, 5, 9, 13, 17)) / 5)
        self.assertEqual(result.confidence, 0.8)

    def test_temporal_selection_does_not_switch_when_hand_scores_cross(self):
        def hand_at(x):
            return [SimpleNamespace(x=x, y=0.5) for _ in range(21)]

        first = SimpleNamespace(hand_landmarks=[hand_at(0.2), hand_at(0.8)], handedness=[[SimpleNamespace(score=0.9)], [SimpleNamespace(score=0.1)]])
        second = SimpleNamespace(hand_landmarks=[hand_at(0.2), hand_at(0.8)], handedness=[[SimpleNamespace(score=0.1)], [SimpleNamespace(score=0.9)]])
        initial = palm_observation(first)
        continued = palm_observation(second, (initial.x, initial.y))
        self.assertAlmostEqual(initial.x, 0.2)
        self.assertAlmostEqual(continued.x, 0.2)
        only_other_hand = SimpleNamespace(hand_landmarks=[hand_at(0.8)], handedness=[[SimpleNamespace(score=0.99)]])
        self.assertIsNone(palm_observation(only_other_hand, (initial.x, initial.y)))

    def test_writes_one_row_per_frame_and_blank_missing_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.mp4"
            make_video(source)
            detector = FakeLandmarker()
            result = process_video(source, csv_output=root / "tracked.csv", annotated_output=root / "annotated.mp4", landmarker=detector)
            with result.csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(result.frame_count, 4)
            self.assertEqual(result.detected_frames, 3)
            self.assertEqual([int(row["frame"]) for row in rows], [0, 1, 2, 3])
            self.assertEqual(detector.timestamps, [0, 100, 200, 300])
            self.assertAlmostEqual(float(rows[0]["hand_x"]), 0.25, places=5)
            self.assertEqual(rows[1]["hand_x"], "")
            self.assertEqual(rows[1]["hand_y"], "")
            self.assertTrue(all(row["hand_z"] == "" for row in rows))
            self.assertTrue(result.annotated_video_path.is_file())

    def test_rejects_outputs_that_alias_source_or_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            make_video(source)
            original = source.read_bytes()
            with self.assertRaisesRegex(ValueError, "source video"):
                process_video(source, annotated_output=source, landmarker=FakeLandmarker())
            self.assertEqual(source.read_bytes(), original)
            output = Path(directory) / "same.mp4"
            with self.assertRaisesRegex(ValueError, "different paths"):
                process_video(source, csv_output=output, annotated_output=output, landmarker=FakeLandmarker())
            hard_link = Path(directory) / "source-hardlink.mp4"
            try:
                hard_link.hardlink_to(source)
            except OSError:
                self.skipTest("Hard links are not supported in this directory")
            with self.assertRaisesRegex(ValueError, "source video"):
                process_video(source, annotated_output=hard_link, landmarker=FakeLandmarker())
            self.assertEqual(source.read_bytes(), original)

    def test_bad_model_file_is_preserved_and_model_cannot_alias_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_model = root / "wrong.task"
            bad_model.write_bytes(b"user data")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                ensure_hand_landmarker_model(bad_model)
            self.assertEqual(bad_model.read_bytes(), b"user data")
            source = root / "source.mp4"
            make_video(source)
            original = source.read_bytes()
            with self.assertRaisesRegex(ValueError, "Model path"):
                process_video(source, annotated_output=root / "annotated.mp4", model_path=source, landmarker=FakeLandmarker())
            self.assertEqual(source.read_bytes(), original)

    def test_uses_decoded_presentation_timestamps_for_vfr_frames(self):
        class TimedCapture:
            positions = [0, 80, 210, 310]

            def __init__(self, _path):
                self.index = 0

            def isOpened(self):
                return True

            def get(self, property_id):
                if property_id == cv2.CAP_PROP_FPS:
                    return 10.0
                if property_id == cv2.CAP_PROP_FRAME_WIDTH:
                    return 64
                if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
                    return 48
                if property_id == cv2.CAP_PROP_POS_MSEC:
                    return self.positions[max(0, self.index - 1)]
                return 0

            def read(self):
                if self.index >= len(self.positions):
                    return False, None
                self.index += 1
                return True, np.zeros((48, 64, 3), dtype=np.uint8)

            def release(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "vfr.mp4"
            source.write_bytes(b"fixture")
            detector = FakeLandmarker()
            with mock.patch("perception.video_processor.cv2.VideoCapture", TimedCapture):
                result = process_video(source, csv_output=root / "tracked.csv", annotated_output=root / "out.mp4", landmarker=detector)
            with result.csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(detector.timestamps, [0, 80, 210, 310])
            np.testing.assert_allclose([float(row["time"]) for row in rows], [0, 0.08, 0.21, 0.31])


if __name__ == "__main__":
    unittest.main()
