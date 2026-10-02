"""Map tracked normalized hand coordinates into a fixed Panda workspace."""

import argparse

from perception.retargeting import WorkspaceMapping, map_hand_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracked_csv")
    parser.add_argument("--output", required=True, help="Output robot XYZ CSV")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--max-missing-fraction", type=float, default=0.2)
    parser.add_argument("--max-speed", type=float, default=0.35, help="Meters per second")
    parser.add_argument("--image-bounds", nargs=4, type=float, metavar=("LEFT", "RIGHT", "TOP", "BOTTOM"), default=(0.2, 0.8, 0.15, 0.85))
    parser.add_argument("--robot-bounds", nargs=4, type=float, metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"), default=(0.38, 0.68, -0.22, 0.22))
    parser.add_argument("--robot-z", type=float, default=0.62)
    args = parser.parse_args()
    mapping = WorkspaceMapping(*args.image_bounds, *args.robot_bounds, args.robot_z)
    trajectory = map_hand_trajectory(args.tracked_csv, args.output, mapping=mapping, confidence_threshold=args.confidence, max_missing_fraction=args.max_missing_fraction, max_speed=args.max_speed)
    print(f"Mapped {len(trajectory)} frames to {args.output}")


if __name__ == "__main__":
    main()
