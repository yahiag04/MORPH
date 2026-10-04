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
from world_model.fidelity import (
    assign_outer_folds, compare_predicted_outcomes, load_episode_outcomes,
    validate_episode_outcomes,
)
from world_model.fidelity import contiguous_episode_indices, rollout_episode, score_terminal_state
from world_model.transition_timing import transition_interval
from world_model.state_dynamics import CONTACT_STATE_INDICES, STATE_GROUPS


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
        count = len(data["states"])
        for key in required:
            if len(data[key]) != count:
                raise ValueError(f"{path} arrays must have matching transition counts")
        for key in ("human_labels", "phases"):
            data[key] = archive[key] if key in archive.files else np.full(count, "", dtype=str)
            if data[key].shape != (count,):
                raise ValueError(f"{path} {key} must have one value per transition")
        for key in ("simulation_start_times", "simulation_end_times"):
            if key in archive.files:
                data[key] = archive[key].astype(np.float64)
                if data[key].shape != (count,):
                    raise ValueError(f"{path} {key} must have one value per transition")
        if "episode_ids" in archive.files:
            data["episode_ids"] = archive["episode_ids"].astype(str)
        data["dt_seconds"] = transition_interval(archive.get("transition_dt_seconds"),
                                                count=len(data["states"]), legacy_interval=legacy_interval)
    data["clip_ids"] = data["clip_ids"].astype(str)
    expected_episode_ids = np.char.add(data["clip_ids"], f":{method}")
    if "episode_ids" in data:
        if data["episode_ids"].shape != (len(data["clip_ids"]),) or not np.array_equal(
            data["episode_ids"], expected_episode_ids
        ):
            raise ValueError(f"{path} episode_ids do not match clip_ids and method {method!r}")
    else:
        data["episode_ids"] = expected_episode_ids
    data["human_labels"] = data["human_labels"].astype(str)
    data["phases"] = data["phases"].astype(str)
    data["method_ids"] = np.full(len(data["clip_ids"]), method, dtype=f"<U{len(method)}")
    return data


def _merge(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = ("states", "actions", "next_states", "clip_ids", "episode_ids",
            "human_labels", "phases", "method_ids")
    timing_keys = ("simulation_start_times", "simulation_end_times")
    if all(all(key in part for key in timing_keys) for part in parts):
        keys += timing_keys
    return {key: np.concatenate([part[key] for part in parts]) for key in keys}


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


def _episode_fidelity(fit, data, transition_dt, common_steps: int):
    """Evaluate complete held-out episodes and return only aggregate-ready rows."""
    if "simulation_start_times" not in data or "simulation_end_times" not in data:
        raise ValueError("full-episode fidelity requires simulation start/end times in both archives")
    sequences = contiguous_episode_indices(
        data["episode_ids"], data["simulation_start_times"], data["simulation_end_times"]
    )
    episodes = [(episode, indices) for episode, indices in sequences
                if str(data["clip_ids"][indices[0]]) not in
                (set(fit.train_clip_ids) | set(fit.validation_clip_ids))]
    if not episodes:
        raise ValueError("outer fold contains no complete held-out episodes")
    horizons = {"one_step": 1, "0.5_seconds": max(1, round(0.5 / transition_dt)),
                "2.5_seconds": max(1, round(2.5 / transition_dt)),
                "longest_common_episode": common_steps}
    if max(horizons.values()) > common_steps:
        raise ValueError("common complete episode is shorter than a requested fidelity horizon")
    values = {name: {group: {"model": [], "persistence": []} for group in STATE_GROUPS}
              for name in horizons}
    contact_predictions = {name: [] for name in horizons}
    contact_truths = {name: [] for name in horizons}
    predictions = []
    uncertainty = []
    position_error = []
    for episode_id, indices in episodes:
        states = data["states"][indices]
        actions = data["actions"][indices]
        truth = data["next_states"][indices]
        predicted, disagreement = rollout_episode(fit.model, fit.normalizer, states[0], actions)
        clip_id = str(data["clip_ids"][indices[0]])
        method = str(data["method_ids"][indices[0]])
        predictions.append({"clip_id": clip_id, "method": method,
                            "score": score_terminal_state(predicted, TASK_CONFIG)})
        for horizon_name, horizon in horizons.items():
            model_state = predicted[horizon]
            target_state = truth[horizon - 1]
            persistence_state = states[0]
            for group, feature_slice in STATE_GROUPS.items():
                values[horizon_name][group]["model"].append(model_state[feature_slice] - target_state[feature_slice])
                values[horizon_name][group]["persistence"].append(persistence_state[feature_slice] - target_state[feature_slice])
            contact_predictions[horizon_name].append(model_state[list(CONTACT_STATE_INDICES)])
            contact_truths[horizon_name].append(target_state[list(CONTACT_STATE_INDICES)])
        uncertainty.extend(disagreement.tolist())
        position_error.extend(np.linalg.norm(predicted[1:, 17:20] - truth[:, 17:20], axis=1).tolist())
    metrics = {}
    for horizon_name, horizon in horizons.items():
        groups = {}
        for group in STATE_GROUPS:
            groups[group] = {
                "world_model_rmse": float(np.sqrt(np.mean(np.square(np.concatenate(values[horizon_name][group]["model"]))))) if values[horizon_name][group]["model"] else None,
                "persistence_rmse": float(np.sqrt(np.mean(np.square(np.concatenate(values[horizon_name][group]["persistence"]))))),
            }
        contacts = np.asarray(contact_predictions[horizon_name]) >= 0.5
        contact_truth = np.asarray(contact_truths[horizon_name]) >= 0.5
        tp = int(np.sum(contacts & contact_truth))
        fp = int(np.sum(contacts & ~contact_truth))
        fn = int(np.sum(~contacts & contact_truth))
        metrics[horizon_name] = {
            "steps": int(horizon), "seconds": round(float(horizon * transition_dt), 6),
            "episode_count": len(episodes), "groups": groups,
            "contact_accuracy": float(np.mean(contacts == contact_truth)),
            "contact_f1": float(2 * tp / max(1, 2 * tp + fp + fn)),
        }
    disagreement_array = np.asarray(uncertainty)
    errors = np.asarray(position_error)
    correlation = (float(np.corrcoef(disagreement_array, errors)[0, 1])
                   if len(errors) > 1 and np.std(disagreement_array) > 0 and np.std(errors) > 0 else None)
    return {"horizons": metrics, "predictions": predictions,
            "disagreement": disagreement_array, "position_error": errors,
            "disagreement_error_pearson": correlation,
            "episode_count": len(episodes), "common_steps": common_steps}


TASK_CONFIG = {
    "tray_inner_half_size_xy": 0.11,
    "package_half_size_xy": (0.025, 0.020),
    "max_linear_speed": 0.02,
    "max_angular_speed": 0.5,
    "minimum_lift_m": 0.05,
}


def _save_fidelity_figure(report: dict, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizon_names = list(report["horizons"])
    seconds = [report["horizons"][name]["seconds"] for name in horizon_names]
    groups = {
        "ee_position": ("End-effector position", "#2b6cb0"),
        "arm_position": ("Arm position", "#2f855a"),
        "package_position": ("Package position", "#805ad5"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for group, (label, color) in groups.items():
        learned = [report["horizons"][name]["groups"][group]["world_model_rmse"] for name in horizon_names]
        baseline = [report["horizons"][name]["groups"][group]["persistence_rmse"] for name in horizon_names]
        std = [report["horizons"][name]["fold_std"]["groups"][group]["world_model_rmse"]
               for name in horizon_names]
        axes[0].errorbar(seconds, learned, yerr=std, color=color, marker="o",
                         linewidth=2, capsize=3, label=f"{label} · model")
        axes[0].plot(seconds, baseline, color=color, linestyle="--", alpha=0.75,
                     label=f"{label} · persistence")
    axes[0].set(xlabel="Prediction horizon (s)", ylabel="Endpoint RMSE", title="Held-out endpoint fidelity")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    density = axes[1].hexbin(report["position_error"], report["disagreement"],
                             gridsize=42, mincnt=1, bins="log", cmap="viridis")
    fig.colorbar(density, ax=axes[1], label="Log step count")
    axes[1].set(xlabel="Package position error (m)", ylabel="Ensemble disagreement",
                title=f"Disagreement vs. error (r={report['disagreement_error_pearson']:.3f})")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-transitions", required=True, type=Path)
    parser.add_argument("--confidence-transitions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--episode-results-dir", required=True, type=Path,
                        help="local contact-evaluation directory with per_clip outcome summaries")
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
    if "simulation_start_times" not in data or "simulation_end_times" not in data:
        parser.error("full-episode fidelity requires simulation start/end times in both archives")
    outcomes = load_episode_outcomes(args.episode_results_dir)
    expected_outcome_keys = set(zip(direct["clip_ids"], direct["method_ids"]))
    expected_outcome_keys.update(zip(confidence["clip_ids"], confidence["method_ids"]))
    validate_episode_outcomes(outcomes, expected_outcome_keys)
    unique_clips = np.unique(data["clip_ids"])
    if len(unique_clips) < 4:
        parser.error("four-fold evaluation requires at least four source clips")
    row_folds = assign_outer_folds(data["clip_ids"], fold_count=4, seed=args.seed)
    complete_sequences = contiguous_episode_indices(
        data["episode_ids"], data["simulation_start_times"], data["simulation_end_times"]
    )
    longest_common_steps = min(len(indices) for _, indices in complete_sequences)
    fold_results = []
    fidelity_folds = []
    all_predictions = []
    for fold_index in range(4):
        test_mask = row_folds == fold_index
        test_clips = np.unique(data["clip_ids"][test_mask])
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
        fidelity = _episode_fidelity(fit, data, transition_dt, longest_common_steps)
        fidelity_folds.append(fidelity)
        all_predictions.extend(fidelity["predictions"])
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
              f"(persistence {result['groups']['ee_position']['persistence_rmse']:.5f} m), "
              f"full-episode clips={fidelity['episode_count']}",
              flush=True)

    group_keys = tuple(fold_results[0]["groups"])
    public_folds = [{key: value for key, value in fold.items()
                     if key not in {"outer_test_clip_ids", "inner_train_clip_ids", "inner_validation_clip_ids"}}
                    for fold in fold_results]
    report = {
        "state_dim": 37, "action_dim": 4, "outer_fold_count": 4,
        "source_clip_count": len(unique_clips), "transition_count": len(data["states"]),
        "rollout_horizon_steps": 25, "rollout_seconds": round(25 * transition_dt, 6),
        "transition_dt_seconds": round(float(transition_dt), 8),
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
    horizon_names = tuple(fidelity_folds[0]["horizons"])
    horizon_report = {}
    for name in horizon_names:
        fold_rows = [fold["horizons"][name] for fold in fidelity_folds]
        metric_groups = {}
        for group in STATE_GROUPS:
            metric_groups[group] = {
                metric: float(np.mean([row["groups"][group][metric] for row in fold_rows]))
                for metric in ("world_model_rmse", "persistence_rmse")
            }
        horizon_report[name] = {
            "steps": fold_rows[0]["steps"], "seconds": fold_rows[0]["seconds"],
            "episode_count": int(sum(row["episode_count"] for row in fold_rows)),
            "groups": metric_groups,
            "contact_accuracy": float(np.mean([row["contact_accuracy"] for row in fold_rows])),
            "contact_f1": float(np.mean([row["contact_f1"] for row in fold_rows])),
            "fold_std": {
                "groups": {group: {
                    metric: float(np.std([row["groups"][group][metric] for row in fold_rows]))
                    for metric in ("world_model_rmse", "persistence_rmse")
                } for group in STATE_GROUPS},
                "contact_accuracy": float(np.std([row["contact_accuracy"] for row in fold_rows])),
                "contact_f1": float(np.std([row["contact_f1"] for row in fold_rows])),
            },
        }
    comparison = compare_predicted_outcomes(all_predictions, outcomes)
    uncertainty = np.concatenate([fold["disagreement"] for fold in fidelity_folds])
    rollout_error = np.concatenate([fold["position_error"] for fold in fidelity_folds])
    correlation = (float(np.corrcoef(uncertainty, rollout_error)[0, 1])
                   if np.std(uncertainty) > 0 and np.std(rollout_error) > 0 else None)
    fidelity_report = {
        "experiment": "human-demonstration-driven MuJoCo dynamics",
        "state_dim": 37, "action_dim": 4, "transition_count": int(len(data["states"])),
        "source_clip_count": int(len(unique_clips)), "outer_fold_count": 4,
        "transition_dt_seconds": round(float(transition_dt), 8), "horizons": horizon_report,
        "candidate_selection": comparison,
        "uncertainty_error_association": {
            "measure": "Pearson correlation across held-out rollout steps",
            "quantity": "ensemble disagreement vs package position error in metres",
            "pearson_r": correlation,
            "step_count": int(len(uncertainty)),
        },
        "evaluation_protocol": "All transitions from each source video, including both retargeting methods, are held out together. Complete contiguous episodes are rolled forward from the first state using recorded actions. Training and residual-gain calibration use training and inner-validation source videos only.",
        "limitations": "Task scores are fractions of six simulator-observable criteria and are not calibrated probabilities. Outcomes are MuJoCo labels, not physical-robot results. The learned model predicts structured state trajectories rather than pixels or raw video.",
    }
    public_dir = ROOT / "results"
    (public_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (public_dir / "metrics" / "world_model_fidelity.json").write_text(
        json.dumps(fidelity_report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _save_fidelity_figure({"horizons": horizon_report, "position_error": rollout_error,
                           "disagreement": uncertainty,
                           "disagreement_error_pearson": correlation},
                          public_dir / "figures" / "world_model_fidelity.png")
    print(f"Saved grouped report and fold checkpoints in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
