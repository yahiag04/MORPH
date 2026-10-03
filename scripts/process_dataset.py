"""Batch process labeled local videos and write the manifest for evaluation."""

import argparse
import json
from pathlib import Path

from perception.video_processor import process_video

VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi"}


def process_dataset(video_root: str | Path, output_dir: str | Path, model_path=None):
    """Process immediate files in labeled folders, keeping outputs outside the repo."""
    root = Path(video_root).expanduser().resolve()
    output = Path(output_dir).expanduser()
    repo_root = Path(__file__).resolve().parents[1]
    resolved_output = output.resolve()
    if resolved_output == repo_root or repo_root in resolved_output.parents:
        raise ValueError("output directory must be outside the repository to keep personal data local")
    if not root.is_dir():
        raise ValueError(f"video root is not a directory: {root}")
    label_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not label_dirs:
        raise ValueError(f"video root must contain labeled subdirectories: {root}")
    videos = [
        (directory.name, video)
        for directory in label_dirs
        for video in sorted(directory.iterdir())
        if video.is_file() and video.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not videos:
        raise ValueError(f"no supported videos found under labeled directories in {root}")
    destinations = [(label.casefold(), video.stem.casefold()) for label, video in videos]
    if len(destinations) != len(set(destinations)):
        raise ValueError("duplicate video names within a label would overwrite local outputs")

    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "processing_manifest.json"
    rows, failures = [], []
    for label, video in videos:
        clip_output = output / "processed" / label
        csv_path = clip_output / f"{video.stem}.csv"
        annotated_path = clip_output / f"{video.stem}_tracked.mp4"
        try:
            result = process_video(
                video,
                csv_output=csv_path,
                annotated_output=annotated_path,
                model_path=model_path,
            )
            rows.append({
                "label": label,
                "source": video.name,
                "csv": str(result.csv_path.absolute()),
                "annotated_video": str(result.annotated_video_path.absolute()),
                "frame_count": result.frame_count,
                "detected_frames": result.detected_frames,
                "fps": result.fps,
                "duration_seconds": result.frame_count / result.fps,
            })
            print(f"OK {label}/{video.name}: {result.detected_frames}/{result.frame_count} frames")
        except Exception as error:
            failures.append({"label": label, "source": video.name, "error": str(error)})
            print(f"FAILED {label}/{video.name}: {error}")

    manifest.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if failures:
        (output / "processing_errors.json").write_text(
            json.dumps(failures, indent=2) + "\n", encoding="utf-8"
        )
    else:
        (output / "processing_errors.json").unlink(missing_ok=True)
    print(f"Processed {len(rows)}/{len(videos)} videos; failures: {len(failures)}")
    print(f"Processing manifest: {manifest}")
    return manifest, rows, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_root", type=Path, help="Directory containing labeled subdirectories")
    parser.add_argument("--output-dir", required=True, type=Path, help="Local output directory outside the repository")
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    try:
        _, _, failures = process_dataset(args.video_root, args.output_dir, args.model_path)
    except ValueError as error:
        parser.error(str(error))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
