"""Held-out dynamics, contact-event and within-scene candidate evaluation."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np

from world_model.v3.contracts import (
    ACTION_DIMS,
    CONTACT_INDICES,
    QUATERNION_SLICE,
    STATE_DIM,
    EpisodeBatch,
    RolloutBatch,
)


STATE_GROUPS = {
    "ee_position": slice(0, 3),
    "arm_position": slice(3, 10),
    "arm_velocity": slice(10, 17),
    "package_position": slice(17, 20),
    "package_linear_velocity": slice(24, 27),
    "package_angular_velocity": slice(27, 30),
    "gripper_aperture": slice(30, 31),
    "gripper_velocity": slice(31, 32),
}


def validate_frozen_protocol(
    protocol: dict[str, Any], *, action_mode: str,
    dataset_manifest_sha256: str, checkpoint_sha256: str,
    test_group_ids: set[str],
) -> None:
    """Bind a final test evaluation to its predeclared dataset and checkpoint."""
    required = {
        "schema_version", "frozen", "action_mode", "dataset_manifest_sha256",
        "checkpoint_sha256", "test_group_ids",
    }
    if not isinstance(protocol, dict) or not required <= protocol.keys():
        raise ValueError("frozen protocol is missing required fields")
    if protocol["schema_version"] != 1 or protocol["frozen"] is not True:
        raise ValueError("final test requires a schema-1 frozen protocol")
    if protocol["action_mode"] != action_mode:
        raise ValueError("frozen protocol action mode does not match evaluation")
    if protocol["dataset_manifest_sha256"] != dataset_manifest_sha256:
        raise ValueError("frozen protocol dataset manifest hash does not match")
    if protocol["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("frozen protocol checkpoint hash does not match")
    groups = protocol["test_group_ids"]
    if (not isinstance(groups, list) or not all(isinstance(group, str) and group for group in groups)
            or len(groups) != len(set(groups)) or set(groups) != test_group_ids):
        raise ValueError("frozen protocol test families do not match the held-out partition")


class BaselineDynamics:
    """Action-independent physical baselines for persistence and free motion."""

    training_group_ids: tuple[str, ...] = ()

    def __init__(self, mode: str):
        if mode not in ("persistence", "constant_velocity", "zero_velocity"):
            raise ValueError("baseline mode must be persistence, constant_velocity, or zero_velocity")
        self.mode = mode

    def rollout(self, initial_states: np.ndarray, actions: np.ndarray) -> RolloutBatch:
        initial_states = np.asarray(initial_states, dtype=np.float32)
        actions = np.asarray(actions)
        if (initial_states.ndim != 2 or initial_states.shape[1] != STATE_DIM
                or actions.ndim != 3 or actions.shape[0] != len(initial_states)):
            raise ValueError("baseline inputs must have shapes [K,37] and [K,H,A]")
        count, horizon = actions.shape[:2]
        states = np.repeat(initial_states[None, :, None, :], horizon + 1, axis=2)
        if self.mode == "constant_velocity":
            elapsed = np.arange(horizon + 1, dtype=np.float32)[None, :, None] * 0.02
            states[0, :, :, 17:20] = (
                initial_states[:, None, 17:20]
                + elapsed * initial_states[:, None, 24:27]
            )
        if self.mode == "zero_velocity":
            states[0, :, :, 10:17] = 0.0
            states[0, :, :, 24:30] = 0.0
            states[0, :, :, 31] = 0.0
        contacts = np.clip(states[:, :, 1:, 32:34], 0.0, 1.0)
        return RolloutBatch(
            states=states, contact_probs=contacts.astype(np.float32),
            valid=np.ones((1, count), dtype=bool),
        )


def quaternion_angular_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return sign-invariant geodesic orientation errors in radians."""
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.ndim < 1 or predicted.shape[-1] != 4:
        raise ValueError("predicted and target quaternions must have matching (..., 4) shapes")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("quaternion inputs must be finite")
    predicted_norm = np.linalg.norm(predicted, axis=-1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=-1, keepdims=True)
    if np.any(predicted_norm < 1e-8) or np.any(target_norm < 1e-8):
        raise ValueError("quaternion inputs must have nonzero norm")
    predicted = predicted / predicted_norm
    target = target / target_norm
    dot = np.clip(np.abs(np.sum(predicted * target, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _f1(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return None if denominator == 0 else float(2 * tp / denominator)


def match_contact_events(
    probabilities: np.ndarray,
    truth: np.ndarray,
    *,
    tolerance_steps: int = 2,
    threshold: float = 0.5,
) -> dict[str, int | float | None]:
    """Match each predicted transition to at most one event within ±tolerance."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(truth)
    if probabilities.ndim != 1 or truth.shape != probabilities.shape or not len(truth):
        raise ValueError("contact probabilities and binary truth must be matching non-empty vectors")
    if not np.isfinite(probabilities).all() or not np.isin(truth, (0, 1, False, True)).all():
        raise ValueError("contact probabilities must be finite and truth must be binary")
    if not isinstance(tolerance_steps, (int, np.integer)) or tolerance_steps < 0:
        raise ValueError("tolerance_steps must be a nonnegative integer")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    predicted = np.flatnonzero(probabilities >= threshold)
    observed = np.flatnonzero(truth.astype(bool))
    available = set(predicted.tolist())
    tp = 0
    for target_index in observed:
        candidates = [index for index in available
                      if abs(index - int(target_index)) <= tolerance_steps]
        if candidates:
            selected = min(candidates, key=lambda index: (abs(index - int(target_index)), index))
            available.remove(selected)
            tp += 1
    fp = len(available)
    fn = len(observed) - tp
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    return {
        "true_positive": int(tp), "false_positive": int(fp),
        "false_negative": int(fn), "predicted_event_count": int(len(predicted)),
        "observed_event_count": int(len(observed)), "support": int(len(observed)),
        "precision": None if precision_denominator == 0 else float(tp / precision_denominator),
        "recall": None if recall_denominator == 0 else float(tp / recall_denominator),
        "f1": _f1(tp, fp, fn),
    }


def _episode_rows(batch: EpisodeBatch) -> list[tuple[str, str, np.ndarray]]:
    count = len(batch.states)
    if batch.states.shape != (count, STATE_DIM) or count == 0:
        raise ValueError(f"states must have shape (N, {STATE_DIM}) and N > 0")
    action_dim = batch.actions.shape[1] if batch.actions.ndim == 2 else -1
    if action_dim not in ACTION_DIMS.values() or batch.actions.shape[0] != count:
        raise ValueError("actions must have one 4D or 8D row per transition")
    if batch.next_states.shape != batch.states.shape:
        raise ValueError("next_states must have the same shape as states")
    for name in ("episode_ids", "group_ids", "phases", "start_times", "end_times"):
        if np.asarray(getattr(batch, name)).shape != (count,):
            raise ValueError(f"{name} must have one value per transition")
    numeric = (batch.states, batch.actions, batch.next_states,
               batch.start_times, batch.end_times)
    if not all(np.isfinite(values).all() for values in numeric):
        raise ValueError("episode states, actions, and times must be finite")
    for name, states in (("states", batch.states), ("next_states", batch.next_states)):
        norms = np.linalg.norm(states[:, QUATERNION_SLICE], axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-3):
            raise ValueError(f"{name} package quaternions must be normalized")
        if not np.isin(states[:, CONTACT_INDICES], (0.0, 1.0)).all():
            raise ValueError(f"{name} contact states must be binary")
    episodes = batch.episode_ids.astype(str)
    groups = batch.group_ids.astype(str)
    if np.any(episodes == "") or np.any(groups == ""):
        raise ValueError("episode and family IDs must be non-empty")
    result = []
    seen: set[str] = set()
    intervals = batch.end_times - batch.start_times
    if np.any(intervals <= 0):
        raise ValueError("episode transitions must have positive duration")
    observed_dt = batch.metadata.get("observation_dt")
    if observed_dt is not None and (
        not np.isfinite(observed_dt) or observed_dt <= 0
        or not np.allclose(intervals, observed_dt, rtol=0.0, atol=1e-8)
    ):
        raise ValueError("transition duration disagrees with the recorded observation interval")
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        if np.any(np.diff(rows) != 1):
            raise ValueError(f"episode {episode!r} is interleaved or split")
        if episode in seen:
            raise ValueError(f"episode {episode!r} is duplicated")
        seen.add(str(episode))
        episode_groups = np.unique(groups[rows])
        if len(episode_groups) != 1:
            raise ValueError(f"episode {episode!r} crosses scenario families")
        if len(rows) > 1:
            if not np.allclose(batch.start_times[rows[1:]], batch.end_times[rows[:-1]],
                               rtol=0.0, atol=1e-8):
                raise ValueError(f"episode {episode!r} contains a temporal gap or overlap")
            if not np.array_equal(batch.next_states[rows[:-1]], batch.states[rows[1:]]):
                raise ValueError(f"episode {episode!r} state transition chain is broken")
        result.append((str(episode), str(episode_groups[0]), rows))
    return result


def _validate_rollout(rollout: RolloutBatch, window_count: int, horizon: int) -> None:
    if not isinstance(rollout, RolloutBatch):
        raise ValueError("model.rollout must return a RolloutBatch")
    if rollout.states.ndim != 4 or rollout.states.shape[1:] != (
        window_count, horizon + 1, STATE_DIM
    ):
        raise ValueError("rollout states must have shape (ensemble, windows, horizon+1, 37)")
    members = rollout.states.shape[0]
    if members < 1 or rollout.contact_probs.shape != (members, window_count, horizon, 2):
        raise ValueError("rollout contact_probs must have shape (ensemble, windows, horizon, 2)")
    if rollout.valid.shape != (members, window_count):
        raise ValueError("rollout valid must have shape (ensemble, windows)")
    valid = np.asarray(rollout.valid, dtype=bool)
    if not valid.any():
        return
    states = rollout.states[valid]
    contacts = rollout.contact_probs[valid]
    if not np.isfinite(states).all() or not np.isfinite(contacts).all():
        raise ValueError("valid model rollouts must be finite")
    if np.any((contacts < 0.0) | (contacts > 1.0)):
        raise ValueError("rollout contact probabilities must be in [0, 1]")
    if not np.allclose(np.linalg.norm(states[..., QUATERNION_SLICE], axis=-1),
                       1.0, rtol=0.0, atol=1e-3):
        raise ValueError("valid model rollouts must contain normalized quaternions")


def _mean_member_state(states: np.ndarray, contact_probs: np.ndarray,
                       valid: np.ndarray, window: int, step: int) -> np.ndarray | None:
    members = np.flatnonzero(valid[:, window])
    if not len(members):
        return None
    values = states[members, window, step].astype(np.float64, copy=True)
    output = values.mean(axis=0)
    quaternions = values[:, QUATERNION_SLICE]
    reference = quaternions[0]
    signs = np.where((quaternions @ reference) < 0.0, -1.0, 1.0)
    quaternion = np.mean(quaternions * signs[:, None], axis=0)
    norm = np.linalg.norm(quaternion)
    output[QUATERNION_SLICE] = reference if norm < 1e-8 else quaternion / norm
    if step > 0:
        output[list(CONTACT_INDICES)] = contact_probs[members, window, step - 1].mean(axis=0)
    return output.astype(np.float32)


def _window_batches(batch: EpisodeBatch, episodes, horizon: int, chunk_size: int = 128):
    windows = []
    for episode, group, rows in episodes:
        for start in range(0, len(rows) - horizon + 1, horizon):
            indices = rows[start:start + horizon]
            windows.append((episode, group, indices))
    for offset in range(0, len(windows), chunk_size):
        chunk = windows[offset:offset + chunk_size]
        initials = np.stack([batch.states[item[2][0]] for item in chunk])
        actions = np.stack([batch.actions[item[2]] for item in chunk])
        yield chunk, initials, actions


def _confusion(probabilities: np.ndarray, truth: np.ndarray) -> dict[str, int | float | None]:
    predicted = np.asarray(probabilities) >= 0.5
    actual = np.asarray(truth).astype(bool)
    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    tn = int(np.count_nonzero(~predicted & ~actual))
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "true_negative": tn,
        "accuracy": float((tp + tn) / max(1, tp + fp + fn + tn)),
        "precision": None if tp + fp == 0 else float(tp / (tp + fp)),
        "recall": None if tp + fn == 0 else float(tp / (tp + fn)),
        "f1": _f1(tp, fp, fn), "prevalence": float(actual.mean()) if actual.size else None,
        "support": int(actual.sum()),
    }


def _group_rmse(errors: dict[str, list[np.ndarray]], name: str) -> float | None:
    values = errors[name]
    if not values:
        return None
    return float(np.sqrt(np.mean(np.square(np.concatenate(values, axis=0),
                                          dtype=np.float64))))


def evaluate_dynamics(
    model,
    episodes: EpisodeBatch,
    *,
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    """Score open-loop trajectories against baselines on complete episodes."""
    if not horizons or any(not isinstance(value, (int, np.integer)) or value < 1
                           for value in horizons):
        raise ValueError("horizons must contain positive integer observation steps")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must not contain duplicates")
    episode_index = _episode_rows(episodes)
    training_groups = getattr(model, "training_group_ids", None)
    if training_groups is None:
        raise ValueError("model checkpoint must expose training family IDs for held-out evaluation")
    overlap = set(map(str, training_groups)) & set(map(str, episodes.group_ids))
    if overlap:
        raise ValueError("held-out episodes overlap model training families")
    dt = float(episodes.metadata.get("observation_dt", np.median(
        episodes.end_times - episodes.start_times
    )))
    result: dict[str, Any] = {"horizons": {}, "episode_count": len(episode_index)}
    for horizon in horizons:
        errors: dict[str, dict[str, list[np.ndarray]]] = {
            model_name: {name: [] for name in STATE_GROUPS}
            for model_name in ("model", "persistence", "constant_velocity", "zero_velocity")
        }
        endpoint_errors: dict[str, dict[str, list[np.ndarray]]] = {
            model_name: {name: [] for name in STATE_GROUPS}
            for model_name in errors
        }
        orientation_errors: dict[str, list[np.ndarray]] = {
            name: [] for name in errors
        }
        endpoint_orientation_errors: dict[str, list[np.ndarray]] = {
            name: [] for name in errors
        }
        endpoint_phases: dict[str, dict[str, list[np.ndarray]]] = {
            name: {} for name in errors
        }
        family_errors: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
            model_name: {group: {} for group in STATE_GROUPS}
            for model_name in errors
        }
        contact_probabilities: list[np.ndarray] = []
        contact_truth: list[np.ndarray] = []
        family_windows: Counter[str] = Counter()
        expected_windows = sum(max(0, (len(rows) - horizon) // horizon + 1)
                               for _, _, rows in episode_index)
        valid_windows = 0
        member_slots = 0
        valid_members = 0
        for chunk, initial, actions in _window_batches(episodes, episode_index, horizon):
            if not len(chunk):
                continue
            rollout = model.rollout(initial, actions)
            _validate_rollout(rollout, len(chunk), horizon)
            valid = np.asarray(rollout.valid, dtype=bool)
            member_slots += int(valid.size)
            valid_members += int(valid.sum())
            valid_windows += int(valid.any(axis=0).sum())
            probs = np.asarray(rollout.contact_probs)
            for window_index, (_, group, rows) in enumerate(chunk):
                family_windows[group] += int(valid[:, window_index].any())
                if not valid[:, window_index].any():
                    continue
                predicted_states = []
                predicted_states.append(initial[window_index].copy())
                for step in range(1, horizon + 1):
                    mean_state = _mean_member_state(rollout.states, probs, valid,
                                                    window_index, step)
                    if mean_state is None:
                        break
                    predicted_states.append(mean_state)
                if len(predicted_states) != horizon + 1:
                    continue
                predicted_path = np.stack(predicted_states)
                truth_path = np.concatenate((initial[window_index:window_index + 1],
                                            episodes.next_states[rows]), axis=0)
                start_state = initial[window_index]
                persistence_path = np.repeat(start_state[None, :], horizon + 1, axis=0)
                constant_velocity_path = persistence_path.copy()
                elapsed = np.arange(horizon + 1, dtype=np.float32)[:, None] * dt
                constant_velocity_path[:, 17:20] = (
                    start_state[17:20][None, :] + elapsed * start_state[24:27][None, :]
                )
                zero_velocity_path = persistence_path.copy()
                zero_velocity_path[:, 10:17] = 0.0
                zero_velocity_path[:, 24:30] = 0.0
                zero_velocity_path[:, 31] = 0.0
                paths = {
                    "model": predicted_path, "persistence": persistence_path,
                    "constant_velocity": constant_velocity_path,
                    "zero_velocity": zero_velocity_path,
                }
                phase = str(episodes.phases[rows[-1]])
                for name, path in paths.items():
                    orientation = quaternion_angular_error(
                        path[1:, QUATERNION_SLICE], truth_path[1:, QUATERNION_SLICE]
                    )
                    orientation_errors[name].append(orientation)
                    endpoint_orientation_errors[name].append(orientation[-1:])
                    for group_name, feature_slice in STATE_GROUPS.items():
                        difference = path[1:, feature_slice] - truth_path[1:, feature_slice]
                        errors[name][group_name].append(difference)
                        endpoint_errors[name][group_name].append(difference[-1:])
                        family_errors[name][group_name].setdefault(group, []).append(difference)
                        endpoint_phases[name].setdefault(phase, {}).setdefault(
                            group_name, []).append(difference[-1:])
                member_contacts = probs[:, window_index]
                member_mask = valid[:, window_index]
                contact_probabilities.append(member_contacts[member_mask, -1].mean(axis=0))
                contact_truth.append(episodes.next_states[rows[-1], list(CONTACT_INDICES)])

        group_metrics: dict[str, dict[str, float | None]] = {}
        for name in STATE_GROUPS:
            group_metrics[name] = {
                "model_rmse": _group_rmse(errors["model"], name),
                "model_endpoint_rmse": _group_rmse(endpoint_errors["model"], name),
                "persistence_rmse": _group_rmse(errors["persistence"], name),
                "constant_velocity_rmse": (
                    _group_rmse(errors["constant_velocity"], name)
                    if name == "package_position" else None
                ),
                "zero_velocity_rmse": (
                    _group_rmse(errors["zero_velocity"], name)
                    if name in ("arm_velocity", "package_linear_velocity",
                                "package_angular_velocity", "gripper_velocity") else None
                ),
            }
        orientation_metric = {
            model_name: {
                "trajectory_mean_radians": float(np.mean(np.concatenate(values)))
                if values else None,
                "endpoint_mean_radians": float(np.mean(np.concatenate(
                    endpoint_orientation_errors[model_name])))
                if endpoint_orientation_errors[model_name] else None,
            }
            for model_name, values in orientation_errors.items()
        }
        phase_metrics = {
            model_name: {
                phase: {
                    group: float(np.sqrt(np.mean(np.square(np.concatenate(values), dtype=np.float64))))
                    for group, values in sorted(groups.items())
                }
                for phase, groups in sorted(by_phase.items())
            }
            for model_name, by_phase in endpoint_phases.items()
        }
        bootstrap_metrics = {}
        per_family_metrics = {}
        for group_name in STATE_GROUPS:
            family_rows = []
            for family in sorted(set().union(*(
                set(family_errors[name][group_name]) for name in family_errors
            ))):
                metrics = {}
                for model_name in family_errors:
                    values = family_errors[model_name][group_name].get(family, [])
                    metrics[f"{model_name}_rmse"] = (
                        float(np.sqrt(np.mean(np.square(np.concatenate(values), dtype=np.float64))))
                        if values else 0.0
                    )
                family_rows.append({"group_id": family, "metrics": metrics})
            per_family_metrics[group_name] = {
                row["group_id"]: row["metrics"] for row in family_rows
            }
            bootstrap_metrics[group_name] = bootstrap_families(
                family_rows, seed=20261004, samples=2000,
            ) if family_rows else None
        if contact_probabilities:
            endpoint_prob = np.stack(contact_probabilities, axis=0)
            endpoint_truth = np.stack(contact_truth, axis=0)
            endpoint_contact = _confusion(endpoint_prob, endpoint_truth)
        else:
            endpoint_contact = {"accuracy": None, "precision": None, "recall": None,
                                "f1": None, "prevalence": None, "support": 0,
                                "true_positive": 0, "false_positive": 0,
                                "false_negative": 0, "true_negative": 0}
        result["horizons"][str(horizon)] = {
            "steps": int(horizon), "seconds": float(horizon * dt),
            "episode_count": sum(len(rows) >= horizon for _, _, rows in episode_index),
            "window_count": expected_windows, "valid_window_count": valid_windows,
            "valid_member_fraction": None if member_slots == 0 else valid_members / member_slots,
            "groups": group_metrics, "orientation": orientation_metric,
            "family_cluster_bootstrap": bootstrap_metrics,
            "per_family_metrics": per_family_metrics,
            "endpoint_rmse_by_phase": phase_metrics,
            "family_count": len(family_windows),
            "endpoint_contacts": endpoint_contact,
            "windows_by_family": dict(sorted(family_windows.items())),
        }

    one_step_valid: dict[str, np.ndarray] = {
        episode: np.zeros(len(rows), dtype=bool) for episode, _, rows in episode_index
    }
    one_step_contact: dict[str, np.ndarray] = {
        episode: np.zeros((len(rows), len(CONTACT_INDICES)), dtype=np.float64)
        for episode, _, rows in episode_index
    }
    episode_rows = {episode: rows for episode, _, rows in episode_index}
    for chunk, initial, actions in _window_batches(episodes, episode_index, 1):
        rollout = model.rollout(initial, actions)
        _validate_rollout(rollout, len(chunk), 1)
        valid = np.asarray(rollout.valid, dtype=bool)
        probabilities = np.asarray(rollout.contact_probs)[:, :, 0]
        for index, (episode, _, rows) in enumerate(chunk):
            members = np.flatnonzero(valid[:, index])
            if not len(members):
                continue
            offset = int(rows[0] - episode_rows[episode][0])
            one_step_valid[episode][offset] = True
            one_step_contact[episode][offset] = probabilities[members, index].mean(axis=0)
    event_report: dict[str, Any] = {}
    contact_names = ("gripper_contact", "support_contact")
    for name in contact_names:
        episode_reports = []
        pooled_tp = pooled_fp = pooled_fn = 0
        for episode in sorted(episode_rows):
            # Build interval events on the original episode timeline. Invalid model
            # rows stay in place and therefore cannot shift later events in time.
            rows = episode_rows[episode]
            contact_column = CONTACT_INDICES[contact_names.index(name)]
            truth_states = episodes.next_states[rows, contact_column].astype(bool)
            truth_previous = episodes.states[rows, contact_column].astype(bool)
            truth_events = truth_states != truth_previous
            predicted_events = np.zeros(len(rows), dtype=np.float64)
            for row_index in range(len(rows)):
                if not one_step_valid[episode][row_index]:
                    continue
                probability = one_step_contact[episode][row_index,
                    CONTACT_INDICES.index(contact_column)]
                current_state = bool(episodes.states[rows[row_index], contact_column])
                if current_state != (probability >= 0.5):
                    predicted_events[row_index] = 1.0
            metric = match_contact_events(predicted_events, truth_events, tolerance_steps=2)
            episode_reports.append(metric)
            pooled_tp += int(metric["true_positive"])
            pooled_fp += int(metric["false_positive"])
            pooled_fn += int(metric["false_negative"])
        event_report[name] = {
            "episodes": len(episode_reports),
            "true_positive": pooled_tp, "false_positive": pooled_fp,
            "false_negative": pooled_fn,
            "f1": _f1(pooled_tp, pooled_fp, pooled_fn),
            "tolerance_steps": 2, "tolerance_seconds": float(2 * dt),
            "support": int(sum(int(item["support"]) for item in episode_reports)),
            "event_definition": "binary contact-state transition between adjacent observations",
            "covered_transition_fraction": float(np.mean([
                np.mean(one_step_valid[episode]) for episode, _, _ in episode_index
            ])) if episode_index else 0.0,
        }
    result["one_step_contacts"] = {
        "episodes": len(episode_index), "events": event_report,
        "event_matching": "one prediction matched to at most one ground-truth change, ±2 observations",
    }
    return result


def evaluate_candidates(
    predictions: list[dict], outcomes: list[dict]
) -> dict[str, Any]:
    """Measure held-out choice quality only among alternatives from one scene."""
    predicted: dict[tuple[str, str], float] = {}
    for row in predictions:
        if not isinstance(row, dict) or not {"group_id", "candidate_id", "score"} <= row.keys():
            raise ValueError("each prediction needs group_id, candidate_id, and score")
        key = (str(row["group_id"]), str(row["candidate_id"]))
        try:
            score = float(row["score"])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("candidate score must be numeric") from error
        if not key[0] or not key[1] or not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("candidate IDs must be non-empty and score must be finite in [0, 1]")
        if key in predicted:
            raise ValueError(f"duplicate predicted candidate {key}")
        predicted[key] = score
    actual: dict[tuple[str, str], bool] = {}
    for row in outcomes:
        if not isinstance(row, dict) or not {"group_id", "candidate_id", "success"} <= row.keys():
            raise ValueError("each outcome needs group_id, candidate_id, and success")
        key = (str(row["group_id"]), str(row["candidate_id"]))
        if type(row["success"]) is not bool:
            raise ValueError("candidate outcome success must be boolean")
        if key in actual:
            raise ValueError(f"duplicate actual candidate outcome {key}")
        actual[key] = row["success"]
    if set(predicted) != set(actual):
        missing = sorted(set(predicted) - set(actual))
        extra = sorted(set(actual) - set(predicted))
        raise ValueError(f"candidate predictions and outcomes do not match: missing={missing}, extra={extra}")
    by_scene: dict[str, list[str]] = {}
    for group, candidate in predicted:
        by_scene.setdefault(group, []).append(candidate)
    informative_pairs = uninformative_pairs = tied_pairs = 0
    correct_pair_total = 0.0
    regrets: list[float] = []
    selection_successes: list[float] = []
    per_scene = []
    for group, candidate_ids in sorted(by_scene.items()):
        if len(candidate_ids) < 2:
            continue
        scores = {candidate: predicted[(group, candidate)] for candidate in candidate_ids}
        labels = {candidate: actual[(group, candidate)] for candidate in candidate_ids}
        group_correct = group_informative = group_uninformative = group_ties = 0
        for first, second in combinations(candidate_ids, 2):
            if labels[first] == labels[second]:
                uninformative_pairs += 1
                group_uninformative += 1
                continue
            informative_pairs += 1
            group_informative += 1
            if np.isclose(scores[first], scores[second], rtol=0.0, atol=1e-12):
                correct_pair_total += 0.5
                group_correct += 0.5
                tied_pairs += 1
                group_ties += 1
            elif (scores[first] > scores[second]) == labels[first]:
                correct_pair_total += 1.0
                group_correct += 1.0
        best_score = max(scores.values())
        selected = [candidate for candidate in candidate_ids
                    if np.isclose(scores[candidate], best_score, rtol=0.0, atol=1e-12)]
        selected_success = float(np.mean([labels[candidate] for candidate in selected]))
        best_available = float(any(labels.values()))
        regret = best_available - selected_success
        regrets.append(regret)
        selection_successes.append(selected_success)
        per_scene.append({
            "group_id": group, "candidate_count": len(candidate_ids),
            "informative_pair_count": group_informative,
            "uninformative_pair_count": group_uninformative,
            "pairwise_tie_count": group_ties,
            "pairwise_ranking_accuracy": (
                None if group_informative == 0 else group_correct / group_informative
            ),
            "selected_candidate_count": len(selected),
            "selected_success_rate": selected_success,
            "best_available_success": best_available,
            "selection_regret": regret,
        })
    return {
        "scene_count": len(by_scene), "evaluated_scene_count": len(per_scene),
        "informative_pair_count": informative_pairs,
        "uninformative_pair_count": uninformative_pairs,
        "pairwise_tie_count": tied_pairs,
        "pairwise_ranking_accuracy": (
            None if informative_pairs == 0 else correct_pair_total / informative_pairs
        ),
        "selected_success_rate": None if not selection_successes else float(np.mean(selection_successes)),
        "selection_regret": None if not regrets else float(np.mean(regrets)),
        "per_scene": per_scene,
        "interpretation": "Pairwise ranking uses only different outcomes within the same initial scene.",
    }


def bootstrap_families(
    rows: list[dict], *, seed: int = 20261004, samples: int = 2000
) -> dict[str, Any]:
    """Bootstrap family aggregates so episodes and variants remain clustered."""
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    if not isinstance(samples, (int, np.integer)) or samples < 1:
        raise ValueError("bootstrap samples must be a positive integer")
    if not rows:
        raise ValueError("bootstrap requires at least one family metric row")
    grouped: dict[str, list[dict[str, float]]] = {}
    metric_names: set[str] | None = None
    for row in rows:
        if not isinstance(row, dict) or not {"group_id", "metrics"} <= row.keys():
            raise ValueError("each bootstrap row needs a family group_id and metrics")
        group = str(row["group_id"])
        metrics = row["metrics"]
        if not group or not isinstance(metrics, dict) or not metrics:
            raise ValueError("family ID and metric mapping must be non-empty")
        keys = set(metrics)
        if metric_names is None:
            metric_names = keys
        if keys != metric_names:
            raise ValueError("all bootstrap rows must contain the same metric names")
        values = {name: float(value) for name, value in metrics.items()}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("bootstrap metrics must be finite; omit unsupported metrics")
        grouped.setdefault(group, []).append(values)
    families = sorted(grouped)
    family_values = np.asarray([
        [np.mean([row[name] for row in grouped[group]]) for name in sorted(metric_names)]
        for group in families
    ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(families), size=(samples, len(families)))
    bootstrap = family_values[choices].mean(axis=1)
    names = sorted(metric_names)
    return {
        "family_count": len(families), "row_count": len(rows),
        "samples": int(samples), "seed": int(seed),
        "mean": {name: float(np.mean(family_values[:, index]))
                 for index, name in enumerate(names)},
        "confidence_interval_95": {
            name: [float(value) for value in np.percentile(bootstrap[:, index], [2.5, 97.5])]
            for index, name in enumerate(names)
        },
        "resampling_unit": "scenario family; all rows and candidate variants remain together",
    }
