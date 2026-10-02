"""Tests for importing candidate demonstration recordings unchanged."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from perception.recording import import_demonstration_video


def write_test_video(path: Path) -> None:
    """Write a short video fixture for metadata parsing tests."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create OpenCV test video")
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()


class RecordingImportTests(unittest.TestCase):
    def test_copies_video_without_reencoding_and_writes_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "phone clip.mp4"
            write_test_video(source)
            original = source.read_bytes()

            video_path, metadata_path = import_demonstration_video(
                source, root / "raw"
            )

            self.assertEqual(video_path.name, "demo_001.mp4")
            self.assertEqual(video_path.read_bytes(), original)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["frame_count"], 5)
            self.assertEqual(metadata["width"], 64)
            self.assertEqual(metadata["height"], 48)
            self.assertAlmostEqual(metadata["fps"], 10.0, places=1)

    def test_rejects_missing_or_unsupported_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "recording"):
                import_demonstration_video(root / "missing.mp4", root / "raw")

            unsupported = root / "notes.txt"
            unsupported.write_text("not a video", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "video format"):
                import_demonstration_video(unsupported, root / "raw")


if __name__ == "__main__":
    unittest.main()
