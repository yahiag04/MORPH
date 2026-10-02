"""Import a smartphone demonstration recording into the local raw-data folder."""

import argparse
from pathlib import Path

from perception.recording import import_demonstration_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Original MP4, MOV, M4V, or AVI")
    parser.add_argument("--name", help="Optional identifier such as demo_001")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    video, metadata = import_demonstration_video(
        args.video,
        project_root / "data/raw",
        demo_name=args.name,
    )
    print(f"Imported original video: {video}")
    print(f"Local metadata: {metadata}")


if __name__ == "__main__":
    main()
