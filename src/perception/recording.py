"""Local import and validation for original demonstration video files."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

import cv2


SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi"}


def _next_demo_name(raw_directory: Path) -> str:
    """Return the next unused sequential demo name in ``raw_directory``."""
    identifiers = []
    for path in raw_directory.glob("demo_*.*"):
        match = re.fullmatch(r"demo_(\d{3,})", path.stem)
        if match:
            identifiers.append(int(match.group(1)))
    return f"demo_{max(identifiers, default=0) + 1:03d}"


def import_demonstration_video(
    source_path: str | Path,
    raw_directory: str | Path,
    *,
    demo_name: str | None = None,
) -> tuple[Path, Path]:
    """Validate a recording, copy it unchanged, and save local video metadata.

    The imported video and JSON sidecar are intended to remain local; the
    repository ignores files placed under ``data/raw`` by default.
    """
    source = Path(source_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Demonstration recording not found: {source}")
    if source.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        raise ValueError(
            f"Unsupported video format '{source.suffix}'; use MP4, MOV, M4V, or AVI"
        )

    capture = cv2.VideoCapture(str(source))
    try:
        opened = capture.isOpened()
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        has_frame, _ = capture.read()
    finally:
        capture.release()
    if not opened or not has_frame or fps <= 0.0 or width <= 0 or height <= 0:
        raise ValueError(f"Could not read video frames and metadata from '{source}'")
    if frame_count <= 0:
        raise ValueError(f"Video '{source}' has no readable frame count")

    destination_dir = Path(raw_directory).expanduser()
    destination_dir.mkdir(parents=True, exist_ok=True)
    name = demo_name or _next_demo_name(destination_dir)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("demo_name may contain only letters, numbers, '_' and '-'")
    destination = destination_dir / f"{name}{source.suffix.lower()}"
    metadata_path = destination_dir / f"{name}.json"
    if destination.exists() or metadata_path.exists():
        raise FileExistsError(f"Demonstration '{name}' already exists in {destination_dir}")

    shutil.copy2(source, destination)
    metadata = {
        "demo_name": name,
        "source_filename": source.name,
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
    }
    try:
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    except OSError:
        destination.unlink(missing_ok=True)
        raise
    return destination, metadata_path
