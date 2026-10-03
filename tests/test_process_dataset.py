"""Tests for labeled local video-batch processing and manifest creation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from perception.video_processor import VideoProcessingResult
from scripts.process_dataset import process_dataset


class ProcessDatasetTests(unittest.TestCase):
    def test_processes_labeled_videos_and_writes_evaluator_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            (videos / "riusciti").mkdir(parents=True)
            (videos / "falliti").mkdir()
            (videos / "riusciti" / "one.MOV").write_bytes(b"video")
            (videos / "falliti" / "two.mp4").write_bytes(b"video")
            (videos / "falliti" / "notes.txt").write_text("ignore")
            (videos / "other" / "nested").mkdir(parents=True)
            (videos / "other" / "nested" / "three.mov").write_bytes(b"video")
            output = root / "local-results"

            def fake_process(video, *, csv_output, annotated_output, model_path):
                return VideoProcessingResult(
                    Path(csv_output), Path(annotated_output), frame_count=10, detected_frames=9, fps=30.0
                )

            with mock.patch("scripts.process_dataset.process_video", side_effect=fake_process) as processor:
                manifest_path, rows, failures = process_dataset(videos, output)

            self.assertEqual(processor.call_count, 2)
            self.assertEqual(failures, [])
            self.assertEqual(manifest_path, output / "processing_manifest.json")
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["label"] for row in manifest}, {"riusciti", "falliti"})
            self.assertTrue(all(Path(row["csv"]).is_relative_to(output) for row in manifest))
            self.assertTrue(all("annotated_video" in row and "frame_count" in row for row in manifest))

    def test_rejects_output_inside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            videos = Path(directory) / "videos"
            (videos / "success").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "outside the repository"):
                process_dataset(videos, Path(__file__).resolve().parents[1])

    def test_rejects_duplicate_stems_that_would_overwrite_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            videos = Path(directory) / "videos" / "success"
            videos.mkdir(parents=True)
            (videos / "take.mov").write_bytes(b"one")
            (videos / "take.mp4").write_bytes(b"two")
            with self.assertRaisesRegex(ValueError, "duplicate video names"):
                process_dataset(videos.parent, Path(directory) / "local-results")


if __name__ == "__main__":
    unittest.main()
