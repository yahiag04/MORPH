#!/usr/bin/env python3
"""Audit V3 randomized-simulation data without loading PyTorch or MuJoCo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from world_model.v3.data import audit_actions, load_partitions


def run_audit(dataset_dir: Path, *, action_modes: tuple[str, ...]) -> dict:
    """Build a local-data digest and aggregate coverage report."""
    root = Path(dataset_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("dataset directory must contain manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 3 or manifest.get("source_kind") != "simulation_randomized":
        raise ValueError("audit requires a schema-3 randomized simulation dataset")
    reports = {}
    for action_mode in action_modes:
        partitions = load_partitions(root, action_mode=action_mode)
        reports[action_mode] = audit_actions(partitions)
    return {
        "schema_version": 1,
        "source_kind": "simulation_randomized",
        "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_episode_count": int(manifest.get("planned_episodes", 0)),
        "manifest_seed": int(manifest["seed"]),
        "action_modes": reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--action-mode", choices=("all", "cartesian4", "actuator8"), default="all")
    args = parser.parse_args(argv)
    output_path = args.output.expanduser().resolve()
    if output_path == ROOT or ROOT in output_path.parents:
        parser.error("--output must remain outside the repository")
    modes = ("cartesian4", "actuator8") if args.action_mode == "all" else (args.action_mode,)
    try:
        report = run_audit(args.dataset_dir, action_modes=modes)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_kind": report["source_kind"],
        "manifest_episode_count": report["manifest_episode_count"],
        "action_modes": sorted(report["action_modes"]),
        "output_file_bytes": output_path.stat().st_size,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
