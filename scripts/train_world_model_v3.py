#!/usr/bin/env python3
"""Train an isolated V3 dynamics ensemble on randomized MuJoCo episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from world_model.v3.contracts import EpisodeBatch
from world_model.v3.data import load_partitions
from world_model.v3.evaluation import evaluate_dynamics
from world_model.v3.training import fit_dynamics


def _select_families(batch: EpisodeBatch, count: int) -> EpisodeBatch:
    families = sorted(np.unique(batch.group_ids.astype(str)))
    if len(families) < count:
        raise ValueError(f"dry run requested {count} families, only {len(families)} are available")
    mask = np.isin(batch.group_ids.astype(str), families[:count])
    return EpisodeBatch(
        **{name: getattr(batch, name)[mask] for name in (
            "states", "actions", "next_states", "episode_ids", "group_ids",
            "phases", "start_times", "end_times",
        )},
        metadata=dict(batch.metadata),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path, action="append",
                        help="dataset root with episodes/; may be repeated for compatible partitions")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--dry-run", action="store_true",
                        help="2 epochs on 4 train and 2 validation families")
    parser.add_argument("--pilot", action="store_true",
                        help="5 epochs on the configured development split")
    parser.add_argument("--resume", action="store_true",
                        help="resume only when checkpoint and run manifest match")
    args = parser.parse_args(argv)
    destination = args.output_dir.expanduser().resolve()
    if destination == ROOT or ROOT in destination.parents:
        parser.error("--output-dir must remain outside the repository")
    if args.dry_run and args.pilot:
        parser.error("--dry-run and --pilot are mutually exclusive")
    try:
        config = json.loads(args.config.expanduser().read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("config root must be a JSON object")
        action_mode = config.get("action_mode")
        partitions = load_partitions(args.dataset_dir, action_mode=action_mode)
        train, validation = partitions["train"], partitions["validation"]
        if args.dry_run:
            train, validation = _select_families(train, 4), _select_families(validation, 2)
            config["epochs"] = 2
            config["rollout_horizon"] = 25
            config["horizon_schedule"] = None
        if args.pilot:
            config["epochs"] = 5
            config["rollout_horizon"] = 25
            config["horizon_schedule"] = None
        if args.seed is not None:
            config["seed"] = args.seed
        if args.max_epochs is not None:
            if args.max_epochs < 1:
                raise ValueError("--max-epochs must be positive")
            config["epochs"] = args.max_epochs
            schedule = config.get("horizon_schedule")
            if schedule and sum(stage.get("epochs", 0) for stage in schedule) != args.max_epochs:
                config["rollout_horizon"] = 25
                config["horizon_schedule"] = None
        model = fit_dynamics(
            train, validation, config=config, output_dir=destination, resume=args.resume,
        )
        horizons = (5, 25, 50, 100)
        report = evaluate_dynamics(model, validation, horizons=horizons)
        (destination / "validation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        event_report = evaluate_dynamics(
            model, validation, horizons=horizons, window_selection="events",
        )
        (destination / "validation_events.json").write_text(
            json.dumps(event_report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, ValueError, ImportError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "status": "complete", "action_mode": model.config.action_mode,
        "training_families": len(model.training_group_ids),
        "validation_families": len(np.unique(validation.group_ids)),
        "output_dir": str(destination), "validation_file": "validation.json",
        "event_validation_file": "validation_events.json",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
