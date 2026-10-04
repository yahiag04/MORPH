#!/usr/bin/env python3
"""Train on assigned scenario families and evaluate the reserved synthetic test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from world_model.state_dynamics import (
    calibrate_rollout_gains, evaluate_state_dynamics, fit_state_dynamics,
    save_checkpoint, validate_transitions,
)
from world_model.transition_timing import transition_interval

ARRAYS = ("states", "actions", "next_states", "episode_ids", "group_ids")
SPLITS = ("train", "validation", "test")


def load_synthetic_partitions(dataset_dir: Path) -> dict[str, dict]:
    """Load complete archives while preserving family-level train/validation/test splits."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    paths = sorted((dataset_dir / "episodes").glob("*.npz"))
    if not paths:
        raise ValueError(f"no episode archives found under {dataset_dir / 'episodes'}")
    collected = {split: {key: [] for key in ARRAYS} for split in SPLITS}
    intervals = set()
    family_splits: dict[str, str] = {}
    seen_episodes: set[str] = set()
    found_splits: set[str] = set()

    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            required = (*ARRAYS, "splits", "source_kinds", "transition_dt_seconds")
            missing = set(required) - set(archive.files)
            if missing:
                raise ValueError(f"{path.name} is missing arrays: {sorted(missing)}")
            rows = len(archive["states"])
            if rows == 0:
                raise ValueError(f"{path.name} is empty")
            split_values = np.unique(archive["splits"].astype(str))
            group_values = np.unique(archive["group_ids"].astype(str))
            episode_values = np.unique(archive["episode_ids"].astype(str))
            source_values = np.unique(archive["source_kinds"].astype(str))
            if any(len(values) != 1 for values in (split_values, group_values, episode_values, source_values)):
                raise ValueError(f"{path.name} contains mixed episode provenance")
            split = str(split_values[0])
            family = str(group_values[0])
            episode = str(episode_values[0])
            if split not in SPLITS:
                raise ValueError(f"{path.name} has unsupported partition {split!r}")
            if source_values[0] != "simulation_randomized":
                raise ValueError(f"{path.name} is not a randomized simulation archive")
            if episode in seen_episodes:
                raise ValueError(f"duplicate episode identifier {episode!r}")
            seen_episodes.add(episode)
            prior_split = family_splits.setdefault(family, split)
            if prior_split != split:
                raise ValueError(f"scenario families cross partitions: {family!r}")
            arrays = {key: archive[key] for key in ARRAYS}
            if any(value.shape[0] != rows for value in arrays.values()):
                raise ValueError(f"{path.name} has inconsistent row counts")
            interval = transition_interval(archive["transition_dt_seconds"], count=rows)
            intervals.add(round(interval, 12))
            for key, value in arrays.items():
                collected[split][key].append(value)
            found_splits.add(split)

    if found_splits != set(SPLITS):
        raise ValueError(f"dataset must contain all partitions: {SPLITS}")
    if len(intervals) != 1:
        raise ValueError("dataset partitions use inconsistent transition intervals")
    result = {}
    for split in SPLITS:
        data = {key: np.concatenate(values) for key, values in collected[split].items()}
        data["states"], data["actions"], data["next_states"] = validate_transitions(
            data["states"], data["actions"], data["next_states"]
        )
        data["episode_ids"] = data["episode_ids"].astype(str)
        data["group_ids"] = data["group_ids"].astype(str)
        data["transition_dt_seconds"] = float(next(iter(intervals)))
        result[split] = data
    if not np.isclose(result["train"]["transition_dt_seconds"],
                      result["validation"]["transition_dt_seconds"]):
        raise ValueError("train and validation intervals differ")
    return result


def evaluate_rollout_baselines(data: dict, *, horizon: int = 25) -> dict:
    """Measure contact persistence and zero-velocity baselines on rollout endpoints."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    state, following = data["states"], data["next_states"]
    episodes = np.asarray(data["episode_ids"]).astype(str)
    if episodes.shape != (len(state),):
        raise ValueError("episode_ids must have one identifier per transition")
    velocity_truth = {"linear": [], "angular": []}
    velocity_persistence = {"linear": [], "angular": []}
    contact_truth, contact_persistence = [], []
    for episode in np.unique(episodes):
        indices = np.flatnonzero(episodes == episode)
        for start in range(0, len(indices) - horizon + 1, horizon):
            window = indices[start:start + horizon]
            initial, truth = state[window[0]], following[window[-1]]
            contact_persistence.append(initial[32:34])
            contact_truth.append(truth[32:34])
            velocity_truth["linear"].append(truth[24:27])
            velocity_persistence["linear"].append(initial[24:27])
            velocity_truth["angular"].append(truth[27:30])
            velocity_persistence["angular"].append(initial[27:30])
    if not contact_truth:
        raise ValueError("no complete rollout windows are available for baseline evaluation")
    predicted_contacts = np.asarray(contact_persistence)
    true_contacts = np.asarray(contact_truth)
    tp = int(np.sum((predicted_contacts == 1) & (true_contacts == 1)))
    fp = int(np.sum((predicted_contacts == 1) & (true_contacts == 0)))
    fn = int(np.sum((predicted_contacts == 0) & (true_contacts == 1)))
    result = {
        "rollouts": len(contact_truth),
        "contact_persistence_accuracy": float(np.mean(predicted_contacts == true_contacts)),
        "contact_persistence_f1": float(2 * tp / max(1, 2 * tp + fp + fn)),
    }
    for name in ("linear", "angular"):
        truth = np.asarray(velocity_truth[name])
        persistence = np.asarray(velocity_persistence[name])
        result[f"package_{name}_velocity"] = {
            "zero_velocity_rmse": float(np.sqrt(np.mean(truth.astype(np.float64) ** 2))),
            "persistence_rmse": float(np.sqrt(np.mean((persistence - truth).astype(np.float64) ** 2))),
        }
    return result


def _public_result(fit, validation_metrics: dict, test_metrics: dict,
                   data: dict[str, dict], args: argparse.Namespace) -> dict:
    return {
        "source_kind": "simulation_randomized",
        "dataset_episode_counts": {split: len(np.unique(data[split]["episode_ids"])) for split in SPLITS},
        "dataset_family_counts": {split: len(np.unique(data[split]["group_ids"])) for split in SPLITS},
        "dataset_transition_counts": {split: len(data[split]["states"]) for split in SPLITS},
        "transition_dt_seconds": fit.transition_dt_seconds,
        "rollout_horizon_steps": test_metrics["rollout_horizon_steps"],
        "rollout_seconds": test_metrics["rollout_seconds"],
        "training": {
            "max_epochs": args.epochs, "epochs_run_per_member": fit.metrics["epochs_per_member"],
            "batch_size": args.batch_size,
            "ensemble_size": args.ensemble_size, "seed": args.seed,
            "train_families_used": len(fit.train_clip_ids),
            "inner_early_stopping_families": len(fit.validation_clip_ids),
            "validation_role": "residual-gain calibration and diagnostic metrics",
            "test_role": "single held-out final evaluation",
        },
        "validation_metrics_after_gain_calibration": validation_metrics,
        "test_metrics": test_metrics,
        "interpretation": "Simulation-only held-out scenario-family performance; this does not establish physical-robot performance.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rollout-horizon", type=int, default=25)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == ROOT or ROOT in output_dir.parents:
        parser.error("--output-dir must be outside a Git repository")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("--output-dir must be empty to protect existing checkpoints")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.ensemble_size < 1:
        parser.error("epochs, batch size, patience and ensemble size must be positive")
    if args.rollout_horizon < 25:
        parser.error("rollout horizon must be at least 25 transitions")

    data = load_synthetic_partitions(args.dataset_dir)
    train, validation, test = (data[split] for split in SPLITS)
    dt = train["transition_dt_seconds"]
    if not all(np.isclose(data[split]["transition_dt_seconds"], dt) for split in SPLITS):
        parser.error("all predefined partitions must have the same transition interval")
    fit = fit_state_dynamics(
        train["states"], train["actions"], train["next_states"], train["group_ids"],
        episode_ids=train["episode_ids"], epochs=args.epochs, batch_size=args.batch_size,
        patience=args.patience, ensemble_size=args.ensemble_size, seed=args.seed,
        rollout_horizon=args.rollout_horizon, transition_dt_seconds=dt,
    )
    calibrated_gains = calibrate_rollout_gains(
        fit.model, fit.normalizer, validation["states"], validation["actions"],
        validation["next_states"], validation["group_ids"], validation["episode_ids"],
        horizon=args.rollout_horizon,
    )
    validation_metrics = evaluate_state_dynamics(
        fit, validation["states"], validation["actions"], validation["next_states"],
        validation["group_ids"], episode_ids=validation["episode_ids"],
        rollout_horizon=args.rollout_horizon, transition_dt_seconds=dt,
    )
    test_metrics = evaluate_state_dynamics(
        fit, test["states"], test["actions"], test["next_states"],
        test["group_ids"], episode_ids=test["episode_ids"],
        rollout_horizon=args.rollout_horizon, transition_dt_seconds=dt,
    )
    test_metrics["simple_rollout_baselines"] = evaluate_rollout_baselines(
        test, horizon=args.rollout_horizon
    )
    fit.model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(fit, str(output_dir / "synthetic_state_dynamics.pt"))
    report = _public_result(fit, validation_metrics, test_metrics, data, args)
    report["calibrated_residual_gains"] = calibrated_gains
    (output_dir / "synthetic_world_model_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Saved checkpoint and synthetic held-out metrics in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
