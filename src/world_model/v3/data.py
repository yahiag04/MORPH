"""Validated, pickle-free V3 transition loaders and action-space audits."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from world_model.v3.contracts import (
    ACTION_DIMS,
    CONTACT_INDICES,
    QUATERNION_SLICE,
    STATE_DIM,
    EpisodeBatch,
)


SYNTHETIC_ARRAYS = (
    "states", "actions", "next_states", "actuator_controls", "episode_ids",
    "group_ids", "splits", "source_kinds", "phases",
    "simulation_start_times", "simulation_end_times", "transition_dt_seconds",
)
HUMAN_ARRAYS = (
    "states", "actions", "next_states", "actuator_controls", "clip_ids",
    "episode_ids", "phases", "simulation_start_times", "simulation_end_times",
    "transition_dt_seconds",
)


def _require_action_mode(action_mode: str) -> int:
    try:
        return ACTION_DIMS[action_mode]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"action_mode must be one of {tuple(ACTION_DIMS)}, got {action_mode!r}"
        ) from error


def _read_archive(path: Path, *, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(set(required) - set(archive.files))
            if missing:
                raise ValueError(f"{path.name} is missing required arrays: {missing}")
            arrays = {name: np.array(archive[name], copy=True) for name in required}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(path.name) in str(error):
            raise
        raise ValueError(f"could not read transition archive {path.name}: {error}") from error
    return arrays


def _check_transition_arrays(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    start_times: np.ndarray,
    end_times: np.ndarray,
    *,
    expected_action_dim: int,
    episode_ids: np.ndarray,
    group_ids: np.ndarray,
) -> None:
    count = len(states)
    if states.shape != (count, STATE_DIM) or count < 2:
        raise ValueError(f"states must have shape (N, {STATE_DIM}) with N >= 2")
    if next_states.shape != states.shape:
        raise ValueError("next_states must have the same shape as states")
    if actions.shape != (count, expected_action_dim):
        raise ValueError(f"actions must have shape (N, {expected_action_dim})")
    if start_times.shape != (count,) or end_times.shape != (count,):
        raise ValueError("simulation timing arrays must have one value per transition")
    if episode_ids.shape != (count,) or group_ids.shape != (count,):
        raise ValueError("episode and group provenance must have one ID per transition")
    if not all(np.isfinite(values).all() for values in
               (states, actions, next_states, start_times, end_times)):
        raise ValueError("transition states, actions, and times must be finite")
    if np.any(end_times <= start_times):
        raise ValueError("every transition must have a positive simulation interval")
    if count > 1 and not np.allclose(start_times[1:], end_times[:-1], rtol=0.0, atol=1e-8):
        raise ValueError("episode timing contains a gap or overlap")
    if count > 1 and not np.array_equal(next_states[:-1], states[1:]):
        raise ValueError("state transition chain is broken: next_states do not meet states")
    for name, values in (("states", states), ("next_states", next_states)):
        norms = np.linalg.norm(values[:, QUATERNION_SLICE], axis=1)
        if not np.isfinite(norms).all() or not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(f"{name} package quaternions must have unit norm")
        if not np.isin(values[:, CONTACT_INDICES], (0.0, 1.0)).all():
            raise ValueError(f"{name} package contact flags must be binary")
    episodes = episode_ids.astype(str)
    groups = group_ids.astype(str)
    if np.any(episodes == "") or np.any(groups == ""):
        raise ValueError("episode and group IDs must be non-empty")
    if len(np.unique(episodes)) != 1 or len(np.unique(groups)) != 1:
        raise ValueError("an archive must contain one contiguous episode and one group")


def _action_array(arrays: dict[str, np.ndarray], action_mode: str) -> np.ndarray:
    if action_mode == "cartesian4":
        return arrays["actions"].astype(np.float32, copy=False)
    return arrays["actuator_controls"].astype(np.float32, copy=False)


def load_partitions(
    dataset_dir: Path, *, action_mode: str
) -> dict[str, EpisodeBatch]:
    """Load synthetic episodes while enforcing episode and family integrity."""
    action_dim = _require_action_mode(action_mode)
    root = Path(dataset_dir).expanduser().resolve()
    episode_paths = sorted((root / "episodes").glob("*.npz"))
    if not episode_paths:
        raise ValueError(f"no episode archives found under {root / 'episodes'}")

    collected: dict[str, dict[str, list[np.ndarray]]] = {
        split: {name: [] for name in (
            "states", "actions", "next_states", "episode_ids", "group_ids",
            "phases", "start_times", "end_times",
        )}
        for split in ("train", "validation", "test")
    }
    family_partitions: dict[str, str] = {}
    seen_episodes: set[str] = set()
    intervals: set[float] = set()
    source_kinds: set[str] = set()

    for path in episode_paths:
        arrays = _read_archive(path, required=SYNTHETIC_ARRAYS)
        states = arrays["states"].astype(np.float32, copy=False)
        next_states = arrays["next_states"].astype(np.float32, copy=False)
        actions = _action_array(arrays, action_mode)
        episodes = arrays["episode_ids"].astype(str)
        groups = arrays["group_ids"].astype(str)
        splits = arrays["splits"].astype(str)
        sources = arrays["source_kinds"].astype(str)
        phases = arrays["phases"].astype(str)
        starts = arrays["simulation_start_times"].astype(np.float64, copy=False)
        ends = arrays["simulation_end_times"].astype(np.float64, copy=False)
        intervals_row = arrays["transition_dt_seconds"].astype(np.float64, copy=False)
        count = len(states)
        if intervals_row.shape != (count,) or not np.isfinite(intervals_row).all() or np.any(intervals_row <= 0):
            raise ValueError(f"{path.name} transition intervals must be positive finite values")
        if not np.allclose(ends - starts, intervals_row, rtol=0.0, atol=1e-8):
            raise ValueError(f"{path.name} transition duration disagrees with simulation times")
        if not np.allclose(intervals_row, intervals_row[0], rtol=0.0, atol=1e-8):
            raise ValueError(f"{path.name} does not use a uniform observation interval")
        for name, values in (("phases", phases), ("splits", splits), ("source_kinds", sources)):
            if values.shape != (count,):
                raise ValueError(f"{path.name} {name} must have one value per transition")
        if len(np.unique(splits)) != 1 or splits[0] not in collected:
            raise ValueError(f"{path.name} must declare exactly one supported partition")
        if len(np.unique(sources)) != 1 or sources[0] != "simulation_randomized":
            raise ValueError(f"{path.name} is not a randomized simulation episode")
        _check_transition_arrays(states, actions, next_states, starts, ends,
                                 expected_action_dim=action_dim, episode_ids=episodes,
                                 group_ids=groups)
        episode, family, split = str(episodes[0]), str(groups[0]), str(splits[0])
        if path.stem != episode:
            raise ValueError(f"{path.name} does not match its episode provenance")
        if episode in seen_episodes:
            raise ValueError(f"duplicate episode ID {episode!r}")
        seen_episodes.add(episode)
        previous = family_partitions.setdefault(family, split)
        if previous != split:
            raise ValueError(f"scenario family {family!r} crosses data partitions")
        intervals.add(round(float(intervals_row[0]), 12))
        source_kinds.add(str(sources[0]))
        target = collected[split]
        target["states"].append(states)
        target["actions"].append(actions)
        target["next_states"].append(next_states)
        target["episode_ids"].append(episodes)
        target["group_ids"].append(groups)
        target["phases"].append(phases)
        target["start_times"].append(starts)
        target["end_times"].append(ends)

    if len(intervals) != 1:
        raise ValueError("all synthetic partitions must use the same observation interval")
    if not collected["train"] or not collected["validation"]:
        raise ValueError("training requires non-empty train and validation partitions")
    dt = next(iter(intervals))
    output: dict[str, EpisodeBatch] = {}
    for split, rows in collected.items():
        if not rows["states"]:
            continue
        output[split] = EpisodeBatch(
            states=np.concatenate(rows["states"]).astype(np.float32, copy=False),
            actions=np.concatenate(rows["actions"]).astype(np.float32, copy=False),
            next_states=np.concatenate(rows["next_states"]).astype(np.float32, copy=False),
            episode_ids=np.concatenate(rows["episode_ids"]).astype(str),
            group_ids=np.concatenate(rows["group_ids"]).astype(str),
            phases=np.concatenate(rows["phases"]).astype(str),
            start_times=np.concatenate(rows["start_times"]).astype(np.float64, copy=False),
            end_times=np.concatenate(rows["end_times"]).astype(np.float64, copy=False),
            metadata={"source_kind": "simulation_randomized", "partition": split,
                      "action_mode": action_mode, "action_dim": action_dim,
                      "observation_dt": dt},
        )
    return output


def load_human_replays(results_dir: Path, *, action_mode: str) -> EpisodeBatch:
    """Read historical paired human replays as diagnostics, never training data."""
    action_dim = _require_action_mode(action_mode)
    root = Path(results_dir).expanduser().resolve()
    archive_paths = tuple(root / f"{method}_transitions.npz"
                          for method in ("direct", "confidence-aware"))
    row_groups: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "states", "actions", "next_states", "episode_ids", "group_ids",
            "phases", "start_times", "end_times",
        )
    }
    clip_to_methods: dict[str, set[str]] = {}
    transition_intervals: set[float] = set()
    for path in archive_paths:
        if not path.is_file():
            raise ValueError(f"missing paired human replay archive: {path.name}")
        arrays = _read_archive(path, required=HUMAN_ARRAYS)
        method = path.name.removesuffix("_transitions.npz")
        clips = arrays["clip_ids"].astype(str)
        episodes = arrays["episode_ids"].astype(str)
        groups = clips.copy()
        sources = arrays.get("source_kinds")
        if sources is not None and np.any(sources.astype(str) != "human_replay"):
            raise ValueError(f"{path.name} contains non-human replay rows")
        if len(clips) != len(arrays["states"]):
            raise ValueError(f"{path.name} clip IDs do not match its transitions")
        expected_episodes = np.char.add(clips, f":{method}")
        if not np.array_equal(episodes, expected_episodes):
            raise ValueError(f"{path.name} episode IDs do not match clips and method")
        actions = _action_array(arrays, action_mode)
        states = arrays["states"].astype(np.float32, copy=False)
        following = arrays["next_states"].astype(np.float32, copy=False)
        starts = arrays["simulation_start_times"].astype(np.float64, copy=False)
        ends = arrays["simulation_end_times"].astype(np.float64, copy=False)
        intervals = arrays["transition_dt_seconds"].astype(np.float64, copy=False)
        if intervals.shape != (len(states),) or not np.isfinite(intervals).all() or np.any(intervals <= 0):
            raise ValueError(f"{path.name} transition intervals must be positive finite values")
        if not np.allclose(ends - starts, intervals, rtol=0.0, atol=1e-8):
            raise ValueError(f"{path.name} transition duration disagrees with simulation times")
        _validate_human_sequences(states, actions, following, starts, ends,
                                  intervals, clips, episodes, action_dim, arrays["phases"])
        transition_intervals.update(round(float(x), 12) for x in intervals)
        for clip in np.unique(clips):
            clip_to_methods.setdefault(str(clip), set()).add(method)
        for name, value in (
            ("states", states), ("actions", actions), ("next_states", following),
            ("episode_ids", episodes), ("group_ids", groups),
            ("phases", arrays["phases"].astype(str)),
            ("start_times", starts), ("end_times", ends),
        ):
            row_groups[name].append(value)
    incomplete = sorted(clip for clip, methods in clip_to_methods.items()
                        if methods != {"direct", "confidence-aware"})
    if incomplete:
        raise ValueError(f"paired human replay methods are missing for source clips: {incomplete}")
    if len(transition_intervals) != 1:
        raise ValueError("paired human replay archives must share one observation interval")
    return EpisodeBatch(
        states=np.concatenate(row_groups["states"]).astype(np.float32, copy=False),
        actions=np.concatenate(row_groups["actions"]).astype(np.float32, copy=False),
        next_states=np.concatenate(row_groups["next_states"]).astype(np.float32, copy=False),
        episode_ids=np.concatenate(row_groups["episode_ids"]).astype(str),
        group_ids=np.concatenate(row_groups["group_ids"]).astype(str),
        phases=np.concatenate(row_groups["phases"]).astype(str),
        start_times=np.concatenate(row_groups["start_times"]).astype(np.float64, copy=False),
        end_times=np.concatenate(row_groups["end_times"]).astype(np.float64, copy=False),
        metadata={"source_kind": "human_replay", "role": "historical_diagnostic",
                  "action_mode": action_mode, "action_dim": action_dim,
                  "observation_dt": next(iter(transition_intervals)),
                  "clip_count": len(clip_to_methods)},
    )


def _validate_human_sequences(
    states: np.ndarray, actions: np.ndarray, next_states: np.ndarray,
    starts: np.ndarray, ends: np.ndarray, intervals: np.ndarray,
    clips: np.ndarray, episodes: np.ndarray, action_dim: int,
    phases: np.ndarray,
) -> None:
    count = len(states)
    if phases.shape != (count,):
        raise ValueError("phases must have one value per human replay transition")
    if actions.shape != (count, action_dim):
        raise ValueError(f"actions must have shape (N, {action_dim})")
    if clips.shape != (count,) or episodes.shape != (count,):
        raise ValueError("clip and episode IDs must have one value per transition")
    if not all(np.isfinite(array).all() for array in
               (states, actions, next_states, starts, ends, intervals)):
        raise ValueError("human replay transitions and times must be finite")
    if np.any(ends <= starts) or np.any(intervals <= 0):
        raise ValueError("human replay transition intervals must be positive")
    if not np.allclose(ends - starts, intervals, rtol=0.0, atol=1e-8):
        raise ValueError("human replay transition intervals disagree with simulation times")
    for name, values in (("states", states), ("next_states", next_states)):
        if values.shape != (count, STATE_DIM):
            raise ValueError(f"{name} must have shape (N, {STATE_DIM})")
        if not np.allclose(np.linalg.norm(values[:, QUATERNION_SLICE], axis=1),
                           1.0, rtol=0.0, atol=1e-3):
            raise ValueError(f"{name} package quaternions must have unit norm")
        if not np.isin(values[:, CONTACT_INDICES], (0.0, 1.0)).all():
            raise ValueError(f"{name} contact flags must be binary")
    seen: set[str] = set()
    for clip in np.unique(clips):
        rows = np.flatnonzero(clips == clip)
        if len(np.unique(episodes[rows])) != 1:
            # Each archive holds a single retargeting method, so an episode is contiguous per clip.
            raise ValueError(f"clip {clip!r} has inconsistent episode IDs")
        current_episode = str(episodes[rows[0]])
        if current_episode in seen:
            raise ValueError(f"episode {current_episode!r} is interleaved")
        seen.add(current_episode)
        if np.any(np.diff(rows) != 1):
            raise ValueError(f"episode {current_episode!r} is interleaved or split")
        if len(rows) > 1:
            if not np.allclose(starts[rows[1:]], ends[rows[:-1]], rtol=0.0, atol=1e-8):
                raise ValueError(f"episode {current_episode!r} contains a timing gap or overlap")
            if not np.array_equal(next_states[rows[:-1]], states[rows[1:]]):
                raise ValueError(f"episode {current_episode!r} state transition chain is broken")


def audit_actions(partitions: dict[str, EpisodeBatch]) -> dict[str, Any]:
    """Summarize command coverage, task phases, contact events and durations."""
    if "train" not in partitions or "validation" not in partitions:
        raise ValueError("action audit requires train and validation partitions")
    action_dim = next(iter(partitions.values())).actions.shape[1]
    report: dict[str, Any] = {"partitions": {}, "action_dim": int(action_dim)}
    for name, batch in partitions.items():
        if batch.actions.ndim != 2 or batch.actions.shape[1] != action_dim:
            raise ValueError("all partitions must contain the same action width")
        episode_ids = batch.episode_ids.astype(str)
        contact_changes = {"gripper_contact": 0, "support_contact": 0}
        durations: list[float] = []
        hold_lengths: list[int] = []
        for episode in np.unique(episode_ids):
            rows = np.flatnonzero(episode_ids == episode)
            if not len(rows) or np.any(np.diff(rows) != 1):
                raise ValueError(f"episode {episode!r} is interleaved in partition {name}")
            durations.append(float(batch.end_times[rows[-1]] - batch.start_times[rows[0]]))
            for index, contact_name in ((32, "gripper_contact"), (33, "support_contact")):
                contact_changes[contact_name] += int(np.count_nonzero(
                    batch.next_states[rows, index] != batch.states[rows, index]
                ))
            commands = batch.actions[rows]
            changed = np.any(commands[1:] != commands[:-1], axis=1)
            boundaries = np.r_[0, np.flatnonzero(changed) + 1, len(rows)]
            hold_lengths.extend(np.diff(boundaries).astype(int).tolist())
        phase_counts = Counter(batch.phases.astype(str).tolist())
        speed = np.linalg.norm(batch.states[:, 24:27], axis=1)
        report["partitions"][name] = {
            "episodes": len(np.unique(episode_ids)),
            "families": len(np.unique(batch.group_ids.astype(str))),
            "transitions": int(len(batch.states)),
            "seconds": float(sum(durations)),
            "episode_duration_seconds": {
                "min": float(min(durations)), "median": float(np.median(durations)),
                "max": float(max(durations)),
            },
            "phase_counts": dict(sorted(phase_counts.items())),
            "contact_changes": contact_changes,
            "moving_object_transitions_over_0_01_mps": int(np.count_nonzero(speed > 0.01)),
            "action_min": batch.actions.min(axis=0).astype(float).tolist(),
            "action_max": batch.actions.max(axis=0).astype(float).tolist(),
            "action_mean": batch.actions.mean(axis=0).astype(float).tolist(),
            "action_hold_run_length_observations": {
                "count": len(hold_lengths),
                "exactly_five_fraction": float(np.mean(np.asarray(hold_lengths) == 5)),
                "median": float(np.median(hold_lengths)), "maximum": int(max(hold_lengths)),
            },
        }
    train = partitions["train"].actions
    validation = partitions["validation"].actions
    report["validation_action_extrapolation_fraction"] = float(np.mean(
        np.any((validation < train.min(axis=0)) | (validation > train.max(axis=0)), axis=1)
    ))
    report["interpretation"] = (
        "Observed command coverage describes the data; it does not prove controller success."
    )
    return report
