"""Map tracked normalized hand coordinates into a fixed Panda workspace."""

import argparse

from perception.retargeting import WorkspaceMapping, map_hand_trajectory
from perception.workspace_calibration import WorkspaceCalibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracked_csv")
    parser.add_argument("--output", required=True, help="Output robot XYZ CSV")
    parser.add_argument("--calibration", help="Per-video marker calibration JSON")
    parser.add_argument("--method", choices=("direct", "confidence-aware"), default="direct")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--max-missing-fraction", type=float, default=0.2)
    parser.add_argument("--max-speed", type=float, default=0.35, help="Meters per second")
    parser.add_argument("--image-bounds", nargs=4, type=float, metavar=("LEFT", "RIGHT", "TOP", "BOTTOM"), default=(0.2, 0.8, 0.15, 0.85))
    parser.add_argument("--robot-bounds", nargs=4, type=float, metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"), default=(0.38, 0.68, -0.22, 0.22))
    parser.add_argument("--robot-z", type=float, default=0.62)
    args = parser.parse_args()
    mapping = WorkspaceMapping(*args.image_bounds, *args.robot_bounds, args.robot_z)
    calibration = WorkspaceCalibration.load(args.calibration) if args.calibration else None
    trajectory = map_hand_trajectory(
        args.tracked_csv,
        args.output,
        mapping=mapping,
        confidence_threshold=args.confidence,
        max_missing_fraction=args.max_missing_fraction,
        max_speed=args.max_speed,
        image_corners=calibration.image_corners if calibration else None,
        method=args.method,
    )
    calibration_source = str(args.calibration) if args.calibration else "legacy rectangular image bounds"
    print(f"Mapped {len(trajectory)} frames to {args.output}")
    print(f"Method: {args.method}; calibration: {calibration_source}")


if __name__ == "__main__":
    main()
