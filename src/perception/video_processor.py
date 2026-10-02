"""Extract normalized palm observations and an annotated local video."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import ssl
import urllib.request

import certifi
import cv2
import mediapipe as mp
import numpy as np

from .hand_tracking import palm_observation


VIDEO_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
VIDEO_MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
CSV_COLUMNS = ("frame", "time", "hand_x", "hand_y", "hand_z", "confidence")


@dataclass(frozen=True)
class VideoProcessingResult:
    csv_path: Path
    annotated_video_path: Path
    frame_count: int
    detected_frames: int
    fps: float


def ensure_hand_landmarker_model(model_path: str | Path) -> Path:
    """Download the official task model atomically if it is not already present."""
    path = Path(model_path).expanduser()
    if path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest == VIDEO_MODEL_SHA256:
            return path
        raise ValueError(f"Existing model file failed SHA-256 verification: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    request = urllib.request.Request(VIDEO_MODEL_URL, headers={"User-Agent": "MORPH/1.0"})
    try:
        with urllib.request.urlopen(
            request, context=ssl.create_default_context(cafile=certifi.where()), timeout=60
        ) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != VIDEO_MODEL_SHA256:
            raise OSError("Downloaded hand tracking model failed SHA-256 verification")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _paths_alias(first: Path, second: Path) -> bool:
    """Compare resolved paths and existing filesystem identities (hard links)."""
    if first == second:
        return True
    try:
        return first.exists() and second.exists() and first.samefile(second)
    except OSError:
        return False


def _create_landmarker(model_path: Path):
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def process_video(
    video_path: str | Path,
    *,
    csv_output: str | Path | None = None,
    annotated_output: str | Path | None = None,
    model_path: str | Path | None = None,
    landmarker=None,
) -> VideoProcessingResult:
    """Track palms on every frame, leaving missing detections and metric Z blank."""
    source = Path(video_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    csv_path = Path(csv_output) if csv_output else Path("data/processed") / f"{source.stem}.csv"
    video_output = Path(annotated_output) if annotated_output else Path("data/processed") / f"{source.stem}_tracked.mp4"
    resolved_source = source.resolve()
    resolved_csv = csv_path.expanduser().resolve()
    resolved_video = video_output.expanduser().resolve()
    if _paths_alias(resolved_source, resolved_csv) or _paths_alias(resolved_source, resolved_video):
        raise ValueError("Output paths must not overwrite the source video")
    if _paths_alias(resolved_csv, resolved_video):
        raise ValueError("CSV and annotated video outputs must use different paths")
    resolved_model = Path(model_path or "assets/models/hand_landmarker.task").expanduser().resolve()
    model_temporary = resolved_model.with_suffix(resolved_model.suffix + ".download")
    if any(_paths_alias(resolved_model, item) for item in (resolved_source, resolved_csv, resolved_video)):
        raise ValueError("Model path must be separate from source and output paths")
    if any(_paths_alias(model_temporary, item) for item in (resolved_source, resolved_csv, resolved_video)):
        raise ValueError("Model download path must be separate from source and output paths")
    csv_path, video_output = resolved_csv, resolved_video
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    video_output.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Video must have positive FPS and frame dimensions")

    owns_landmarker = landmarker is None
    if owns_landmarker:
        actual_model = ensure_hand_landmarker_model(resolved_model)
        landmarker = _create_landmarker(actual_model)
    writer = cv2.VideoWriter(
        str(video_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        if owns_landmarker:
            landmarker.close()
        raise OSError(f"Could not create annotated video: {video_output}")

    frame_count = detected_count = 0
    previous_xy = None
    previous_time = -1.0
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer_csv = csv.writer(stream, lineterminator="\n")
            writer_csv.writerow(CSV_COLUMNS)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoder_time = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                nominal_time = frame_count / fps
                timestamp_seconds = decoder_time if np.isfinite(decoder_time) and decoder_time > previous_time else nominal_time
                if timestamp_seconds <= previous_time:
                    timestamp_seconds = previous_time + 1.0 / fps
                timestamp_ms = round(timestamp_seconds * 1000.0)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                observation = palm_observation(landmarker.detect_for_video(mp_image, timestamp_ms), previous_xy)
                if observation:
                    detected_count += 1
                    previous_xy = (observation.x, observation.y)
                    writer_csv.writerow((frame_count, f"{timestamp_seconds:.6f}", f"{observation.x:.7f}", f"{observation.y:.7f}", "", f"{observation.confidence:.6f}"))
                    cv2.putText(frame, f"palm {observation.x:.3f}, {observation.y:.3f}  conf {observation.confidence:.2f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 2)
                    cv2.circle(frame, (round(observation.x * width), round(observation.y * height)), max(5, width // 100), (40, 220, 40), -1)
                else:
                    writer_csv.writerow((frame_count, f"{timestamp_seconds:.6f}", "", "", "", "0.000000"))
                    cv2.putText(frame, "hand not detected", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 230), 2)
                writer.write(frame)
                frame_count += 1
                previous_time = timestamp_seconds
    finally:
        capture.release()
        writer.release()
        if owns_landmarker:
            landmarker.close()
    if frame_count == 0:
        raise ValueError(f"Video contains no decodable frames: {source}")
    return VideoProcessingResult(csv_path, video_output, frame_count, detected_count, fps)
