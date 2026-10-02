"""Extract a local hand trajectory and annotated video from a recording."""

import argparse

from perception.video_processor import process_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to a local source recording")
    parser.add_argument("--csv", help="Output hand landmark CSV")
    parser.add_argument("--annotated", help="Output annotated MP4")
    parser.add_argument("--model", help="Local MediaPipe model path; downloaded if absent")
    args = parser.parse_args()
    result = process_video(args.video, csv_output=args.csv, annotated_output=args.annotated, model_path=args.model)
    print(f"Tracked {result.detected_frames}/{result.frame_count} frames at {result.fps:g} FPS")
    print(f"CSV: {result.csv_path}")
    print(f"Annotated video: {result.annotated_video_path}")


if __name__ == "__main__":
    main()
