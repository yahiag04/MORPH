"""Calibrate the four pink workspace markers for each labeled video."""

import argparse
from pathlib import Path

import cv2

from perception.workspace_calibration import WorkspaceCalibration, detect_workspace_corners


VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi"}
CORNER_NAMES = ("TL", "TR", "BR", "BL")


def calibrate_video(video_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Save one video's normalized marker calibration and review image."""
    repo_root = Path(__file__).resolve().parents[1]
    resolved_output = output_dir.expanduser().resolve()
    if resolved_output == repo_root or repo_root in resolved_output.parents:
        raise ValueError("output directory must be outside the repository to keep personal data local")
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {video_path}")
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ValueError(f"video has no readable first frame: {video_path}")

    corners = detect_workspace_corners(frame)
    calibration = WorkspaceCalibration(corners)
    height, width = frame.shape[:2]
    points = (corners * (width, height)).round().astype(int)
    preview = frame.copy()
    cv2.polylines(preview, [points.reshape(-1, 1, 2)], True, (0, 255, 255), max(3, width // 500))
    for name, (x, y) in zip(CORNER_NAMES, points, strict=True):
        cv2.circle(preview, (int(x), int(y)), max(8, width // 120), (0, 255, 255), -1)
        cv2.putText(
            preview,
            name,
            (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.8, width / 1800),
            (0, 0, 0),
            max(2, width // 700),
            cv2.LINE_AA,
        )

    target_dir = output_dir / video_path.parent.name
    target_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = target_dir / f"{video_path.stem}.json"
    preview_path = target_dir / f"{video_path.stem}_markers.jpg"
    if not cv2.imwrite(str(preview_path), preview):
        raise OSError(f"could not write marker review image: {preview_path}")
    calibration.save(calibration_path)
    return calibration_path, preview_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_root", type=Path, help="Directory containing labeled subdirectories")
    parser.add_argument("--output-dir", required=True, type=Path, help="Local directory for calibration JSON and review images")
    args = parser.parse_args()

    video_root = args.video_root.expanduser()
    output_dir = args.output_dir.expanduser()
    if not video_root.is_dir():
        parser.error(f"video root is not a directory: {video_root}")
    label_dirs = sorted(path for path in video_root.iterdir() if path.is_dir())
    if not label_dirs:
        parser.error(f"video root must contain labeled subdirectories: {video_root}")
    videos = [
        video
        for label_dir in label_dirs
        for video in sorted(label_dir.iterdir())
        if video.is_file() and video.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not videos:
        parser.error(f"no supported videos found under labeled directories in {video_root}")
    destinations = [(video.parent.name.casefold(), video.stem.casefold()) for video in videos]
    if len(destinations) != len(set(destinations)):
        parser.error("duplicate video names within a label would overwrite local calibration outputs")

    failures = 0
    for video in videos:
        try:
            calibration_path, preview_path = calibrate_video(video, output_dir)
        except (OSError, ValueError, cv2.error) as error:
            failures += 1
            print(f"FAILED {video.parent.name}/{video.name}: {error}")
        else:
            print(f"OK {video.parent.name}/{video.name}: {calibration_path} | review: {preview_path}")
    print(f"Calibrated {len(videos) - failures}/{len(videos)} videos; failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
