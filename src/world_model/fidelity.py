"""Private episode alignment utilities for held-out model-fidelity evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

METHODS = ("direct", "confidence-aware")


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
