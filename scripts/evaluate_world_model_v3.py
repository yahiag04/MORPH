#!/usr/bin/env python3
"""Evaluate a V3 checkpoint on a named, family-isolated data partition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from world_model.v3.data import load_partitions
from world_model.v3.evaluation import evaluate_dynamics, validate_frozen_protocol


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--partition", choices=("validation", "test"), required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--frozen-protocol", type=Path,
                        help="required for test partition; binds dataset, checkpoint, and families")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--action-mode", choices=("cartesian4", "actuator8"), required=True)
    parser.add_argument("--horizons", default="5,25,50,100")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("--output must remain outside the repository")
    try:
        horizons = tuple(int(value.strip()) for value in args.horizons.split(","))
        partitions = load_partitions(args.dataset_dir, action_mode=args.action_mode)
        if args.partition not in partitions:
            raise ValueError(f"dataset has no {args.partition!r} partition")
        if args.partition == "test" and args.frozen_protocol is None:
            raise ValueError("--frozen-protocol is required for final test evaluation")
        if args.partition != "test" and args.frozen_protocol is not None:
            raise ValueError("--frozen-protocol is only valid for final test evaluation")
        if args.partition == "test":
            manifest_path = args.dataset_dir.expanduser().resolve() / "manifest.json"
            protocol = json.loads(args.frozen_protocol.expanduser().read_text(encoding="utf-8"))
            validate_frozen_protocol(
                protocol, action_mode=args.action_mode,
                dataset_manifest_sha256=_sha256(manifest_path),
                checkpoint_sha256=_sha256(args.checkpoint.expanduser()),
                test_group_ids=set(map(str, partitions["test"].group_ids)),
            )
        # Import lazily so --help and data-only tooling do not require PyTorch.
        from world_model.v3.model import DynamicsV3

        model = DynamicsV3.load(args.checkpoint, expected={"action_mode": args.action_mode})
        report = evaluate_dynamics(
            model, partitions[args.partition], horizons=horizons,
        )
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "partition": args.partition, "action_mode": args.action_mode,
        "episode_count": report["episode_count"], "horizons": list(horizons),
        "output_file_bytes": output.stat().st_size,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
