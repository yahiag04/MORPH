"""Private episode alignment utilities for held-out model-fidelity evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

METHODS = ("direct", "confidence-aware")

from .state_dynamics import (
    ACTION_DIM,
    CONTACT_STATE_INDICES,
    QUATERNION_STATE_SLICE,
    STATE_DIM,
    _normalizer_tensors,
    _step_torch,
    torch,
)


def load_episode_outcomes(results_dir: Path) -> dict[tuple[str, str], dict]:
    """Load local per-video MuJoCo outcomes, keyed by source clip and method."""
    root = Path(results_dir).expanduser().resolve()
    summaries = sorted((root / "per_clip").glob("*/*/*_summary.json"))
    if not summaries:
        raise ValueError(f"no per-episode summaries found under {root / 'per_clip'}")

    outcomes: dict[tuple[str, str], dict] = {}
    for path in summaries:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read episode summary '{path.name}'") from error
        if not isinstance(summary, dict):
            raise ValueError(f"episode summary '{path.name}' must contain a JSON object")
        required = {"clip_id", "method", "human_label", "robot_success"}
        missing = required - set(summary)
        if missing:
            raise ValueError(f"episode summary '{path.name}' is missing fields: {sorted(missing)}")

        clip_id = str(summary["clip_id"])
        method = str(summary["method"])
        human_label = str(summary["human_label"])
        if not clip_id or method not in METHODS or not human_label:
            raise ValueError(f"episode summary '{path.name}' has invalid clip/method/human label")
        if not isinstance(summary["robot_success"], bool):
            raise ValueError(f"episode summary '{path.name}' robot_success must be boolean")
        if path.parent.name != clip_id or path.parent.parent.name != human_label:
            raise ValueError(f"episode summary path does not match its clip and human label: '{path.name}'")
        if path.name != f"{method}_summary.json":
            raise ValueError(f"episode summary filename does not match method for '{path.name}'")

        key = (clip_id, method)
        if key in outcomes:
            raise ValueError(f"duplicate episode outcome for clip {clip_id!r}, method {method!r}")
        outcomes[key] = summary
    return outcomes


def validate_episode_outcomes(
    outcomes: dict[tuple[str, str], dict], expected_episode_keys,
) -> None:
    """Require one and only one private MuJoCo result for every archived episode."""
    expected = {(str(clip), str(method)) for clip, method in expected_episode_keys}
    actual = set(outcomes)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"episode outcome mismatch: missing={missing}, unexpected={unexpected}"
        )


def assign_outer_folds(
    clip_ids: np.ndarray, *, fold_count: int = 4, seed: int = 17
) -> np.ndarray:
    """Assign each transition to a deterministic source-clip fold."""
    clips = np.asarray(clip_ids).astype(str)
    if clips.ndim != 1 or clips.size == 0 or np.any(clips == ""):
        raise ValueError("clip_ids must be a non-empty vector of non-empty identifiers")
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    unique = np.unique(clips)
    if len(unique) < fold_count:
        raise ValueError(f"at least {fold_count} unique source clips are required")
    permuted = np.random.default_rng(seed).permutation(unique)
    clip_to_fold = {
        str(clip): fold
        for fold, fold_clips in enumerate(np.array_split(permuted, fold_count))
        for clip in fold_clips
    }
    return np.fromiter((clip_to_fold[clip] for clip in clips), dtype=np.int64, count=len(clips))


def rollout_episode(model, normalizer, initial_state: np.ndarray,
                    actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Autoregressively predict a complete episode and per-step ensemble disagreement.

    Disagreement averages the member spread of normalized continuous deltas and
    contact probabilities. Returned states include the initial observation.
    """
    if torch is None:
        raise ImportError("world-model rollout requires the optional PyTorch runtime")
    try:
        initial = np.asarray(initial_state, dtype=np.float32)
        action_array = np.asarray(actions, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("initial_state and actions must be numeric arrays") from error
    if initial.shape != (STATE_DIM,):
        raise ValueError(f"initial_state must have shape ({STATE_DIM},)")
    if action_array.ndim != 2 or action_array.shape[1] != ACTION_DIM or len(action_array) < 1:
        raise ValueError(f"actions must have shape (T, {ACTION_DIM}) with T >= 1")
    if not np.isfinite(initial).all() or not np.isfinite(action_array).all():
        raise ValueError("initial_state and actions must contain only finite values")
    if not np.isclose(np.linalg.norm(initial[QUATERNION_STATE_SLICE]), 1.0, atol=1e-3):
        raise ValueError("initial_state package quaternion must have unit norm")
    if not np.isin(initial[list(CONTACT_STATE_INDICES)], (0.0, 1.0)).all():
        raise ValueError("initial_state contact flags must be binary")

    model_was_training = bool(model.training)
    model.eval()
    states = np.empty((len(action_array) + 1, STATE_DIM), dtype=np.float32)
    disagreements = np.empty(len(action_array), dtype=np.float32)
    states[0] = initial
    params = _normalizer_tensors(normalizer)
    with torch.no_grad():
        current = torch.as_tensor(initial.reshape(1, -1))
        for index, action in enumerate(action_array):
            action_tensor = torch.as_tensor(action.reshape(1, -1))
            next_state, _, _ = _step_torch(
                model, current, action_tensor, params, hard_contacts=True
            )
            normalized_state = (current - params["state_mean"]) / params["state_scale"]
            normalized_action = (action_tensor - params["action_mean"]) / params["action_scale"]
            member_outputs = [
                member(normalized_state, normalized_action) for member in model.members
            ]
            continuous = torch.stack([output[0] for output in member_outputs])
            contact_probability = torch.stack([
                torch.sigmoid(output[1]) for output in member_outputs
            ])
            continuous_spread = continuous.std(dim=0, unbiased=False).mean()
            contact_spread = contact_probability.std(dim=0, unbiased=False).mean()
            disagreements[index] = float(((continuous_spread + contact_spread) / 2.0).item())
            current = next_state
            states[index + 1] = current[0].cpu().numpy()
    if model_was_training:
        model.train()
    if not np.isfinite(states).all() or not np.isfinite(disagreements).all():
        raise ValueError("predicted rollout contains non-finite values")
    if not np.allclose(np.linalg.norm(states[:, QUATERNION_STATE_SLICE], axis=1), 1.0, atol=1e-5):
        raise ValueError("predicted rollout contains a non-unit package quaternion")
    if not np.isin(states[:, CONTACT_STATE_INDICES], (0.0, 1.0)).all():
        raise ValueError("predicted rollout contains non-binary contact flags")
    return states, disagreements


def score_terminal_state(states: np.ndarray, task_config: dict) -> float:
    """Score physical pick-and-place criteria as an uncalibrated 0-to-1 fraction.

    The score averages six observable conditions: gripper contact occurred,
    at least the requested lift was reached, package footprint is inside the
    tray, support contact is present, and both linear and angular speed are below
    the simulation's stability limits. It is a ranking score, not a probability
    or a replacement for the episode's robot_success label.
    """
    trajectory = np.asarray(states, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[0] < 2 or trajectory.shape[1] != STATE_DIM:
        raise ValueError(f"states must have shape (T, {STATE_DIM}) with T >= 2")
    if not np.isfinite(trajectory).all():
        raise ValueError("states must contain only finite values")
    if not np.allclose(np.linalg.norm(trajectory[:, QUATERNION_STATE_SLICE], axis=1), 1.0, atol=1e-3):
        raise ValueError("states package quaternions must have unit norm")
    if not np.isin(trajectory[:, CONTACT_STATE_INDICES], (0.0, 1.0)).all():
        raise ValueError("states contact flags must be binary")
    required = {
        "tray_inner_half_size_xy", "package_half_size_xy", "max_linear_speed",
        "max_angular_speed", "minimum_lift_m",
    }
    if required - set(task_config):
        raise ValueError(f"task_config is missing values: {sorted(required - set(task_config))}")
    initial_z = trajectory[0, 19]
    terminal = trajectory[-1]
    package_xyz = terminal[17:20]
    tray_xyz = terminal[34:37]
    package_half_x, package_half_y = task_config["package_half_size_xy"]
    usable_x = float(task_config["tray_inner_half_size_xy"]) - float(package_half_x)
    usable_y = float(task_config["tray_inner_half_size_xy"]) - float(package_half_y)
    inside = (abs(package_xyz[0] - tray_xyz[0]) <= usable_x
              and abs(package_xyz[1] - tray_xyz[1]) <= usable_y)
    lifted = float(np.max(trajectory[:, 19]) - initial_z) >= float(task_config["minimum_lift_m"])
    supported = terminal[33] >= 0.5
    grasped = np.any(trajectory[:, 32] >= 0.5)
    linear_stable = np.linalg.norm(terminal[24:27]) <= float(task_config["max_linear_speed"])
    angular_stable = np.linalg.norm(terminal[27:30]) <= float(task_config["max_angular_speed"])
    return float(np.mean((grasped, lifted, inside, supported, linear_stable, angular_stable)))


def compare_predicted_outcomes(predictions: list[dict], outcomes: dict) -> dict:
    """Compare held-out model ranking scores with actual paired MuJoCo outcomes.

    Scores are physical-condition fractions, not probabilities. The reported
    absolute score/outcome gap is therefore descriptive and is not called a
    calibration error.
    """
    prediction_by_key: dict[tuple[str, str], float] = {}
    for row in predictions:
        try:
            key = (str(row["clip_id"]), str(row["method"]))
            score = float(row["score"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("each prediction needs clip_id, method, and numeric score") from error
        if key in prediction_by_key:
            raise ValueError(f"duplicate predicted score for clip {key[0]!r}, method {key[1]!r}")
        if key[1] not in METHODS or not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("predicted score must be finite in [0, 1] for a supported method")
        prediction_by_key[key] = score

    validate_episode_outcomes(outcomes, prediction_by_key)
    if not prediction_by_key:
        raise ValueError("at least one predicted episode score is required")
    labels = {}
    for key in prediction_by_key:
        if key not in outcomes or not isinstance(outcomes[key].get("robot_success"), bool):
            raise ValueError(f"missing boolean MuJoCo outcome for episode {key}")
        labels[key] = outcomes[key]["robot_success"]

    by_clip: dict[str, list[tuple[str, float, bool]]] = {}
    for (clip, method), score in prediction_by_key.items():
        by_clip.setdefault(clip, []).append((method, score, labels[(clip, method)]))
    if any({method for method, _, _ in rows} != set(METHODS) for rows in by_clip.values()):
        raise ValueError("candidate comparison requires both retargeting methods for every clip")

    pair_results = []
    selection_regrets = []
    for rows in by_clip.values():
        success_scores = [score for _, score, success in rows if success]
        failure_scores = [score for _, score, success in rows if not success]
        if success_scores and failure_scores:
            success_score = float(np.mean(success_scores))
            failure_score = float(np.mean(failure_scores))
            pair_results.append(
                0.5 if np.isclose(success_score, failure_score, atol=1e-12)
                else float(success_score > failure_score)
            )
        best_predicted = max(score for _, score, _ in rows)
        selected_success_rate = float(np.mean([
            success for _, score, success in rows
            if np.isclose(score, best_predicted, atol=1e-12)
        ]))
        best_actual = float(any(success for _, _, success in rows))
        selection_regrets.append(best_actual - selected_success_rate)

    gaps = [abs(prediction_by_key[key] - float(labels[key])) for key in prediction_by_key]
    ties = sum(
        int(np.isclose(rows[0][1], rows[1][1], atol=1e-12))
        for rows in by_clip.values()
    )
    return {
        "matched_episode_count": len(prediction_by_key),
        "source_clip_count": len(by_clip),
        "informative_pair_count": len(pair_results),
        "pairwise_tie_count": ties,
        "pairwise_ranking_accuracy": float(np.mean(pair_results)) if pair_results else None,
        "mean_selection_regret": float(np.mean(selection_regrets)),
        "mean_absolute_score_outcome_gap": float(np.mean(gaps)),
        "score_interpretation": "fraction of six physical task criteria; not a calibrated probability",
    }
