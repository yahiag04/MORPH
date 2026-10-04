#!/usr/bin/env python3
"""Run four-fold source-clip-held-out world-model evaluation."""

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
    load_checkpoint, save_checkpoint,
)
from world_model.transition_timing import transition_interval


def _load_pair(path: Path, method: str, legacy_interval: float | None = None) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        required = ("states", "actions", "next_states", "clip_ids")
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        if (("splits" in archive and np.any(archive["splits"] != "unassigned")) or
                ("source_kinds" in archive and np.any(archive["source_kinds"] != "human_replay"))):
            raise ValueError("use the predefined scenario partitions for synthetic data; video splitting is not applicable")
        data = {key: archive[key] for key in required}
        data["dt_seconds"] = transition_interval(archive.get("transition_dt_seconds"),
                                                count=len(data["states"]), legacy_interval=legacy_interval)
    data["clip_ids"] = data["clip_ids"].astype(str)
    data["episode_ids"] = np.char.add(data["clip_ids"], f":{method}")
    return data


def _merge(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts])
            for key in ("states", "actions", "next_states", "clip_ids", "episode_ids")}


def _aggregate(folds: list[dict], keys: tuple[str, ...]) -> dict:
    output = {}
    for group in keys:
        values = [fold["groups"][group] for fold in folds]
        if all(isinstance(value, (int, float)) for value in values):
            output[group] = float(np.mean(values))
            continue
        names = set.intersection(*(set(value) for value in values))
        output[group] = {metric: float(np.mean([value[metric] for value in values]))
                         for metric in sorted(names)
                         if all(value[metric] is not None for value in values)}
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-transitions", required=True, type=Path)
    parser.add_argument("--confidence-transitions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--transition-dt-seconds", type=float,
                        help="explicit interval for legacy archives without timing metadata")
    parser.add_argument("--checkpoint-dt-seconds", type=float,
                        help="explicit training interval for legacy checkpoints without timing metadata")
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == ROOT or ROOT in output_dir.parents:
        parser.error("--output-dir must be outside the repository")
    output_dir.mkdir(parents=True, exist_ok=True)

    direct = _load_pair(args.direct_transitions.expanduser(), "direct", args.transition_dt_seconds)
    confidence = _load_pair(args.confidence_transitions.expanduser(), "confidence-aware", args.transition_dt_seconds)
    if not np.isclose(direct["dt_seconds"], confidence["dt_seconds"], rtol=1e-7, atol=1e-9):
        parser.error("both transition archives must have the same observation interval")
    transition_dt = direct["dt_seconds"]
    data = _merge((direct, confidence))
    unique_clips = np.unique(data["clip_ids"])
    if len(unique_clips) < 4:
        parser.error("four-fold evaluation requires at least four source clips")
    folds = np.array_split(np.random.default_rng(args.seed).permutation(unique_clips), 4)
    fold_results = []
    for fold_index, test_clips in enumerate(folds):
        test_mask = np.isin(data["clip_ids"], test_clips)
        train_mask = ~test_mask
        checkpoint_path = output_dir / f"fold_{fold_index + 1}_state_dynamics.pt"
        if checkpoint_path.is_file():
            fit = load_checkpoint(str(checkpoint_path), legacy_interval=args.checkpoint_dt_seconds)
        else:
            fit = fit_state_dynamics(
                data["states"][train_mask], data["actions"][train_mask],
                data["next_states"][train_mask], data["clip_ids"][train_mask],
                episode_ids=data["episode_ids"][train_mask], epochs=args.epochs,
                seed=args.seed + fold_index,
                transition_dt_seconds=transition_dt,
            )
        inner_validation_mask = np.isin(data["clip_ids"][train_mask], fit.validation_clip_ids)
        calibrate_rollout_gains(
            fit.model, fit.normalizer,
            data["states"][train_mask][inner_validation_mask],
            data["actions"][train_mask][inner_validation_mask],
            data["next_states"][train_mask][inner_validation_mask],
            data["clip_ids"][train_mask][inner_validation_mask],
            data["episode_ids"][train_mask][inner_validation_mask],
        )
        save_checkpoint(fit, str(checkpoint_path))
        result = evaluate_state_dynamics(
            fit, data["states"][test_mask], data["actions"][test_mask],
            data["next_states"][test_mask], data["clip_ids"][test_mask],
            episode_ids=data["episode_ids"][test_mask],
            transition_dt_seconds=transition_dt,
        )
        fold_results.append({
            "fold": fold_index + 1,
            "outer_test_clip_ids": sorted(test_clips.tolist()),
            "inner_train_clip_ids": list(fit.train_clip_ids),
            "inner_validation_clip_ids": list(fit.validation_clip_ids),
            **result,
        })
        (output_dir / "world_model_v2_cv_progress.json").write_text(
            json.dumps(fold_results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"fold {fold_index + 1}/4: clips={len(test_clips)}, "
              f"transitions={result['transition_count']}, "
              f"EE rollout={result['groups']['ee_position']['world_model_rmse']:.5f} m "
              f"(persistence {result['groups']['ee_position']['persistence_rmse']:.5f} m)",
              flush=True)

    group_keys = tuple(fold_results[0]["groups"])
    public_folds = [{key: value for key, value in fold.items()
                     if key not in {"outer_test_clip_ids", "inner_train_clip_ids", "inner_validation_clip_ids"}}
                    for fold in fold_results]
    report = {
        "state_dim": 37, "action_dim": 4, "outer_fold_count": 4,
        "source_clip_count": len(unique_clips), "transition_count": len(data["states"]),
        "rollout_horizon_steps": 25, "rollout_seconds": 25 * transition_dt,
        "transition_dt_seconds": transition_dt,
        "fold_mean_groups": _aggregate(fold_results, group_keys),
        "fold_std_groups": {
            group: ({metric: float(np.std([fold["groups"][group][metric] for fold in fold_results]))
                     for metric in fold_results[0]["groups"][group]}
                    if isinstance(fold_results[0]["groups"][group], dict)
                    else float(np.std([fold["groups"][group] for fold in fold_results])))
            for group in group_keys
        },
        "total_outer_test_rollouts": int(sum(fold["groups"]["rollouts"] for fold in fold_results)),
        "folds": public_folds,
        "evaluation_protocol": "Four outer folds grouped by source video; each fold uses a separate grouped inner validation split for early stopping. Direct and confidence-aware rows for the same source video stay together.",
        "limitations": "Simulation-derived transitions; results are a proxy for task dynamics and do not establish physical-robot performance.",
    }
    (output_dir / "world_model_v2_cv.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved grouped report and fold checkpoints in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
