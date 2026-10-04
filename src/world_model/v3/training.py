"""Family-bootstrapped training for the V3 action-conditioned dynamics ensemble."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from world_model.v3.contracts import ACTION_DIMS, CONTACT_INDICES, DynamicsConfig, STATE_DIM
from world_model.v3.evaluation import _episode_rows
from world_model.v3.model import DynamicsNormalizerV3, DynamicsV3, torch

if torch is not None:
    from torch.nn import functional as F


DEFAULTS: dict[str, Any] = {
    "ensemble_size": 3,
    "hidden_dim": 128,
    "seed": 17,
    "epochs": 40,
    "batch_size": 64,
    "learning_rate": 5e-4,
    "weight_decay": 1e-5,
    "gradient_clip": 1.0,
    "patience": 8,
    "rollout_horizon": 25,
    "window_stride": 5,
    "relative_features": False,
    "event_fraction": 0.0,
    "balance_contacts": False,
    "loss_mode": "grouped_state",
    "state_loss_weights": {
        "end_effector_position": 1.0, "arm_position": 1.0, "arm_velocity": 1.0,
        "package_position": 1.0, "package_orientation": 1.0,
        "package_linear_velocity": 1.0, "package_angular_velocity": 1.0,
        "gripper_aperture": 1.0, "gripper_velocity": 1.0,
    },
    "horizon_schedule": None,
    "device": "cpu",
    "observation_dt": 0.02,
    "action_mode": "actuator8",
}
TRAINING_SCHEMA_VERSION = 2


def _state_features(states: np.ndarray, relative_features: bool) -> np.ndarray:
    if not relative_features:
        return np.asarray(states, dtype=np.float32)
    package_from_ee = states[:, 17:20] - states[:, 0:3]
    tray_from_package = states[:, 34:37] - states[:, 17:20]
    return np.concatenate((states, package_from_ee, tray_from_package), axis=1).astype(np.float32)


def fit_normalizer(
    train: EpisodeBatch, *, relative_features: bool = False
) -> DynamicsNormalizerV3:
    """Fit every mean and scale on training transitions only."""
    if len(train.states) < 2 or train.actions.ndim != 2:
        raise ValueError("normalizer requires at least two training transitions")
    features = _state_features(train.states, relative_features)
    delta = train.next_states[:, :32] - train.states[:, :32]
    state_mean, state_scale = features.mean(axis=0), features.std(axis=0)
    action_mean, action_scale = train.actions.mean(axis=0), train.actions.std(axis=0)
    delta_mean, delta_scale = delta.mean(axis=0), delta.std(axis=0)
    state_scale = np.maximum(state_scale, 0.02)
    action_scale = np.maximum(action_scale, 1e-5)
    delta_scale = np.maximum(delta_scale, 0.01)
    delta_scale[20:24] = np.maximum(delta_scale[20:24], 0.01)
    return DynamicsNormalizerV3(*(
        value.astype(np.float32) for value in (
            state_mean, state_scale, action_mean, action_scale, delta_mean, delta_scale,
        )
    ))


def _torch_features(state, relative_features: bool):
    if not relative_features:
        return state
    return torch.cat((state, state[:, 17:20] - state[:, 0:3],
                      state[:, 34:37] - state[:, 17:20]), dim=-1)


def _step(member, state, action, config: dict, params: dict):
    features = _torch_features(state, config["relative_features"])
    normalized_state = (features - params["state_mean"]) / params["state_scale"]
    normalized_action = (action - params["action_mean"]) / params["action_scale"]
    normalized_delta, contact_logits = member(normalized_state, normalized_action)
    delta = params["delta_mean"] + normalized_delta * params["delta_scale"]
    dynamic = state[:, :32] + delta
    quaternion = dynamic[:, 20:24]
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    fallback = state[:, 20:24]
    quaternion = torch.where(norm > 1e-8, quaternion / norm.clamp_min(1e-8), fallback)
    dynamic = torch.cat((dynamic[:, :20], quaternion, dynamic[:, 24:]), dim=-1)
    contact_probs = torch.sigmoid(contact_logits)
    next_state = torch.cat((dynamic, contact_probs, state[:, 34:37]), dim=-1)
    return next_state, normalized_delta, contact_logits, contact_probs


_STATE_LOSS_GROUPS = (
    ("end_effector_position", slice(0, 3), 1.0),
    ("arm_position", slice(3, 10), 1.0),
    ("arm_velocity", slice(10, 17), 0.5),
    ("package_position", slice(17, 20), 3.0),
    ("package_orientation", slice(20, 24), 1.0),
    ("package_linear_velocity", slice(24, 27), 1.0),
    ("package_angular_velocity", slice(27, 30), 1.0),
    ("gripper_aperture", slice(30, 31), 1.0),
    ("gripper_velocity", slice(31, 32), 0.5),
)


def _weighted_state_loss(predicted, truth, state_scale, group_weights=None):
    """Train-normalized physical state loss with a sign-invariant quaternion term."""
    weights = group_weights or {name: weight for name, _, weight in _STATE_LOSS_GROUPS}
    losses = []
    total_weight = 0.0
    for name, columns, default_weight in _STATE_LOSS_GROUPS:
        weight = float(weights.get(name, default_weight))
        if weight <= 0 or not np.isfinite(weight):
            raise ValueError(f"state loss weight {name} must be finite and positive")
        if name == "package_orientation":
            pred_q = F.normalize(predicted[:, columns], dim=-1, eps=1e-8)
            true_q = F.normalize(truth[:, columns], dim=-1, eps=1e-8)
            dot = torch.sum(pred_q * true_q, dim=-1).clamp(-1.0, 1.0)
            error = 1.0 - dot.square()
        else:
            scale = state_scale[columns].clamp_min(1e-6)
            error = (((predicted[:, columns] - truth[:, columns]) / scale) ** 2).mean(dim=-1)
        losses.append(error.mean() * weight)
        total_weight += weight
    return torch.stack(losses).sum() / total_weight


def _contact_class_weights(train: EpisodeBatch) -> tuple[np.ndarray, dict]:
    """Compute clipped per-head negative/positive ratios from training rows only."""
    values = np.asarray(train.next_states[:, CONTACT_INDICES], dtype=np.int64)
    weights, report = [], {}
    names = ("gripper_contact", "support_contact")
    for index, name in enumerate(names):
        positives = int(values[:, index].sum())
        negatives = int(len(values) - positives)
        weight = 1.0 if not positives or not negatives else float(np.clip(negatives / positives, 0.5, 10.0))
        weights.append(weight)
        report[name] = {"positives": positives, "negatives": negatives,
                        "positive_weight": weight,
                        "missing_class": "positive" if not positives else
                                        "negative" if not negatives else None}
    return np.asarray(weights, dtype=np.float32), report


def _contact_loss(logits, truth, positive_weight):
    return F.binary_cross_entropy_with_logits(
        logits, truth[:, list(CONTACT_INDICES)], pos_weight=positive_weight,
    )


def _state_prediction_loss(predicted, truth, state_scale, config):
    if config.get("loss_mode", "grouped_state") == "delta_mse":
        error = (predicted[:, :32] - truth[:, :32]) / state_scale[:32].clamp_min(1e-6)
        return torch.mean(error.square())
    return _weighted_state_loss(
        predicted, truth, state_scale, config.get("state_loss_weights"),
    )


def _horizon_for_epoch(schedule: tuple[tuple[int, int], ...], epoch: int) -> int:
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    cursor = 0
    for horizon, epochs in schedule:
        if epoch < cursor + epochs:
            return horizon
        cursor += epochs
    raise ValueError("epoch falls outside the configured horizon schedule")


def _window_loss(
    member, states, actions, next_states, config, params, *, train: bool,
    return_components: bool = False,
):
    horizon = states.shape[1]
    checkpoints = {1, min(5, horizon), horizon}
    one_step_state, one_step_contact = [], []
    autoregressive_state, autoregressive_contact = [], []
    current = states[:, 0]
    for offset in range(states.shape[1]):
        truth = next_states[:, offset]
        observed_prediction, observed_delta, observed_logits, _ = _step(
            member, states[:, offset], actions[:, offset], config, params,
        )
        if config.get("loss_mode", "grouped_state") == "delta_mse":
            target_delta = (
                truth[:, :32] - states[:, offset, :32] - params["delta_mean"]
            ) / params["delta_scale"]
            observed_state_loss = F.mse_loss(observed_delta, target_delta)
        else:
            observed_state_loss = _state_prediction_loss(
                observed_prediction, truth, params["state_scale"], config,
            )
        observed_contact_loss = _contact_loss(
            observed_logits, truth, params["contact_pos_weight"],
        )
        one_step_state.append(observed_state_loss)
        one_step_contact.append(observed_contact_loss)

        prediction, _, rollout_logits, _ = _step(
            member, current, actions[:, offset], config, params,
        )
        current = prediction
        if offset + 1 in checkpoints:
            autoregressive_state.append(_state_prediction_loss(
                current, truth, params["state_scale"], config,
            ))
            autoregressive_contact.append(
                _contact_loss(rollout_logits, truth, params["contact_pos_weight"])
            )
    one_state = torch.stack(one_step_state).mean()
    one_contact = torch.stack(one_step_contact).mean()
    auto_state = torch.stack(autoregressive_state).mean()
    auto_contact = torch.stack(autoregressive_contact).mean()
    observed = one_state + one_contact
    imagined = auto_state + auto_contact
    total = 0.5 * observed + 0.5 * imagined
    if return_components:
        return total, {"total": total, "one_step_state": one_state,
                       "one_step_contact": one_contact, "autoregressive_state": auto_state,
                       "autoregressive_contact": auto_contact,
                       "one_step": observed, "autoregressive": imagined}
    return total


def _make_windows(batch: EpisodeBatch, horizon: int, stride: int) -> list[tuple[str, str, np.ndarray]]:
    windows = []
    for episode, family, rows in _episode_rows(batch):
        for start in range(0, len(rows) - horizon + 1, stride):
            windows.append((episode, family, rows[start:start + horizon]))
    return windows


def _event_windows(batch: EpisodeBatch, horizon: int) -> list[tuple[str, str, np.ndarray]]:
    windows = []
    for episode, family, rows in _episode_rows(batch):
        latest_start = len(rows) - horizon
        if latest_start < 0:
            continue
        events = np.flatnonzero(np.any(
            batch.states[rows][:, list(CONTACT_INDICES)]
            != batch.next_states[rows][:, list(CONTACT_INDICES)], axis=1,
        ))
        for event in events:
            first_start = max(0, int(event) - 5)
            last_start = min(int(event), latest_start)
            if first_start <= last_start:
                for start in range(first_start, last_start + 1):
                    windows.append((episode, family, rows[start:start + horizon]))
    return windows


def _sampling_pools(batch: EpisodeBatch, horizon: int, stride: int) -> dict:
    uniform_windows = _make_windows(batch, horizon, stride)
    event_windows = _event_windows(batch, horizon)
    uniform_by_episode: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    event_by_episode: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    episode_family = {}
    for window in uniform_windows:
        uniform_by_episode.setdefault(window[0], []).append(window)
        episode_family[window[0]] = window[1]
    for window in event_windows:
        event_by_episode.setdefault(window[0], []).append(window)
        episode_family[window[0]] = window[1]
    episodes_by_family: dict[str, list[str]] = {}
    for episode, family in episode_family.items():
        if episode in uniform_by_episode:
            episodes_by_family.setdefault(family, []).append(episode)
    return {"uniform_windows": uniform_windows, "event_windows": event_windows,
            "uniform_by_episode": uniform_by_episode, "event_by_episode": event_by_episode,
            "episodes_by_family": episodes_by_family}


def _sample_training_windows(
    batch: EpisodeBatch, *, horizon: int, stride: int, count: int,
    rng: np.random.Generator, event_fraction: float,
    family_draws: np.ndarray | None = None,
    pools: dict | None = None,
) -> tuple[list[tuple[str, str, np.ndarray]], dict]:
    if count < 1 or not 0.0 <= event_fraction <= 1.0:
        raise ValueError("sample count must be positive and event_fraction must be in [0,1]")
    pools = pools or _sampling_pools(batch, horizon, stride)
    uniform_windows = pools["uniform_windows"]
    event_windows = pools["event_windows"]
    uniform_by_episode = pools["uniform_by_episode"]
    event_by_episode = pools["event_by_episode"]
    episodes_by_family = pools["episodes_by_family"]
    family_ids = np.asarray(sorted(episodes_by_family), dtype=str)
    eligible_family_draws = family_ids if family_draws is None else np.asarray(
        [family for family in family_draws if family in episodes_by_family], dtype=str,
    )
    if not len(eligible_family_draws):
        raise ValueError("training family sample must refer to families with uniform windows")
    requested_events = int(np.floor(count * event_fraction))
    selected, sampled_events, fallback = [], 0, 0
    for sample_index in range(count):
        family = str(rng.choice(eligible_family_draws))
        episode = str(rng.choice(episodes_by_family[family]))
        use_event = sample_index < requested_events
        choices = event_by_episode.get(episode, []) if use_event else []
        if choices:
            selected.append(choices[int(rng.integers(len(choices)))])
            sampled_events += 1
        else:
            choices = uniform_by_episode[episode]
            selected.append(choices[int(rng.integers(len(choices)))])
            if use_event:
                fallback += 1
    return selected, {
        "requested_event_windows": requested_events,
        "sampled_event_windows": sampled_events,
        "uniform_fallback_windows": fallback,
        "event_candidate_count": len(event_windows),
    }


def _batch_tensors(batch: EpisodeBatch, index: np.ndarray, device):
    return tuple(torch.as_tensor(value[index], dtype=torch.float32, device=device) for value in (
        batch.states, batch.actions, batch.next_states,
    ))


def _validation_score(member, batch, windows, config, params, batch_size, device):
    by_family: dict[str, list[float]] = {}
    with torch.no_grad():
        for family in sorted({item[1] for item in windows}):
            family_windows = [item[2] for item in windows if item[1] == family]
            losses = []
            for start in range(0, len(family_windows), batch_size):
                selected = family_windows[start:start + batch_size]
                idx = np.stack(selected)
                state_rows = torch.as_tensor(batch.states[idx], dtype=torch.float32, device=device)
                action_rows = torch.as_tensor(batch.actions[idx], dtype=torch.float32, device=device)
                next_rows = torch.as_tensor(batch.next_states[idx], dtype=torch.float32, device=device)
                losses.append((float(_window_loss(
                    member, state_rows, action_rows, next_rows, config, params, train=False,
                ).item()), len(selected)))
            total_windows = sum(count for _, count in losses)
            by_family[family] = [sum(value * count for value, count in losses) / total_windows]
    return float(np.mean([np.mean(values) for values in by_family.values()])), by_family


def _batch_hash(batch: EpisodeBatch) -> str:
    digest = hashlib.sha256()
    for name in ("states", "actions", "next_states", "episode_ids", "group_ids",
                 "phases", "start_times", "end_times"):
        values = np.ascontiguousarray(getattr(batch, name))
        digest.update(name.encode("utf-8"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _source_hash() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _resolve_config(config: dict[str, Any], train: EpisodeBatch) -> dict[str, Any]:
    resolved = {**DEFAULTS, **config}
    if resolved["action_mode"] not in ACTION_DIMS:
        raise ValueError("action_mode must be cartesian4 or actuator8")
    if train.actions.shape[1] != ACTION_DIMS[resolved["action_mode"]]:
        raise ValueError("training action dimension does not match action_mode")
    positive = ("ensemble_size", "hidden_dim", "epochs", "batch_size", "patience",
                "rollout_horizon", "window_stride")
    if any(not isinstance(resolved[name], int) or resolved[name] < 1 for name in positive):
        raise ValueError("ensemble, hidden, epoch, batch, patience and horizon values must be positive integers")
    if resolved["rollout_horizon"] < 25:
        raise ValueError("rollout_horizon must be at least 25 for the V2-style rollout objective")
    if not np.isfinite(resolved["event_fraction"]) or not 0.0 <= resolved["event_fraction"] <= 1.0:
        raise ValueError("event_fraction must be finite and in [0,1]")
    if not isinstance(resolved["balance_contacts"], bool):
        raise ValueError("balance_contacts must be boolean")
    if resolved["loss_mode"] not in ("grouped_state", "delta_mse"):
        raise ValueError("loss_mode must be grouped_state or delta_mse")
    weights = resolved["state_loss_weights"]
    if weights is None:
        resolved["state_loss_weights"] = dict(DEFAULTS["state_loss_weights"])
        weights = resolved["state_loss_weights"]
    if weights is not None:
        known = {name for name, _, _ in _STATE_LOSS_GROUPS}
        if not isinstance(weights, dict) or set(weights) - known:
            raise ValueError("state_loss_weights must be a mapping of known groups")
        if any(not np.isfinite(value) or value <= 0 for value in weights.values()):
            raise ValueError("state_loss_weights values must be finite and positive")
    raw_schedule = resolved["horizon_schedule"]
    if raw_schedule is None:
        schedule = [{"horizon": resolved["rollout_horizon"], "epochs": resolved["epochs"]}]
    else:
        if not isinstance(raw_schedule, list) or not raw_schedule:
            raise ValueError("horizon_schedule must be a nonempty list of horizon/epochs stages")
        schedule = []
        for stage in raw_schedule:
            if not isinstance(stage, dict) or set(stage) != {"horizon", "epochs"}:
                raise ValueError("each horizon_schedule stage needs exactly horizon and epochs")
            if any(not isinstance(stage[key], int) or stage[key] < 1 for key in stage):
                raise ValueError("horizon schedule values must be positive integers")
            if stage["horizon"] < 25:
                raise ValueError("curriculum horizons must be at least 25 observations")
            schedule.append({"horizon": stage["horizon"], "epochs": stage["epochs"]})
        if schedule[0]["horizon"] != resolved["rollout_horizon"]:
            raise ValueError("rollout_horizon must equal the first horizon curriculum stage")
        if sum(stage["epochs"] for stage in schedule) != resolved["epochs"]:
            raise ValueError("horizon curriculum epochs must sum to configured epochs")
    resolved["horizon_schedule"] = schedule
    for name in ("learning_rate", "weight_decay", "gradient_clip", "observation_dt"):
        if not np.isfinite(resolved[name]) or resolved[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if resolved["device"] not in ("cpu", "mps"):
        raise ValueError("device must be cpu or mps")
    if resolved["device"] == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    if train.metadata.get("observation_dt") is not None and not np.isclose(
        train.metadata["observation_dt"], resolved["observation_dt"], rtol=0.0, atol=1e-8,
    ):
        raise ValueError("training observation interval does not match config")
    return resolved


def fit_dynamics(
    train: EpisodeBatch, validation: EpisodeBatch, *, config: dict,
    output_dir: Path, resume: bool = False,
) -> DynamicsV3:
    """Fit and checkpoint an action-conditioned ensemble on isolated families."""
    if torch is None:
        raise ImportError("V3 model training requires the optional PyTorch runtime")
    resolved = _resolve_config(config, train)
    if validation.actions.shape[1] != ACTION_DIMS[resolved["action_mode"]]:
        raise ValueError("validation action dimension does not match action_mode")
    train_episodes = _episode_rows(train)
    validation_episodes = _episode_rows(validation)
    train_groups = {family for _, family, _ in train_episodes}
    validation_groups = {family for _, family, _ in validation_episodes}
    if train_groups & validation_groups:
        raise ValueError("training and validation scenario families overlap")
    if train.metadata.get("source_kind") != "simulation_randomized" or validation.metadata.get(
        "source_kind"
    ) != "simulation_randomized":
        raise ValueError("Task 3 fitting accepts randomized simulation data only")
    if not np.isclose(train.metadata.get("observation_dt", resolved["observation_dt"]),
                      validation.metadata.get("observation_dt", resolved["observation_dt"]),
                      rtol=0.0, atol=1e-8):
        raise ValueError("train and validation observation intervals differ")
    if not train_groups or not validation_groups:
        raise ValueError("train and validation partitions must both be non-empty")
    schedule = tuple((stage["horizon"], stage["epochs"])
                     for stage in resolved["horizon_schedule"])
    horizons = sorted({horizon for horizon, _ in schedule})
    train_pools = {horizon: _sampling_pools(train, horizon, resolved["window_stride"])
                   for horizon in horizons}
    validation_windows = {
        horizon: _make_windows(validation, horizon, resolved["window_stride"])
        for horizon in horizons
    }
    if any(not train_pools[horizon]["uniform_windows"] or not validation_windows[horizon]
           for horizon in horizons):
        raise ValueError("train and validation episodes need full windows for every curriculum horizon")
    train_window_counts = {str(horizon): len(train_pools[horizon]["uniform_windows"])
                           for horizon in horizons}
    validation_window_counts = {str(horizon): len(validation_windows[horizon])
                                for horizon in horizons}

    destination = Path(output_dir).expanduser().resolve()
    resolved["seed"] = int(resolved["seed"])
    contact_weight_values, contact_weight_report = _contact_class_weights(train)
    if not resolved["balance_contacts"]:
        contact_weight_values[:] = 1.0
        for value in contact_weight_report.values():
            value["positive_weight"] = 1.0
    run_manifest = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "action_mode": resolved["action_mode"],
        "action_dim": ACTION_DIMS[resolved["action_mode"]],
        "state_dim": STATE_DIM,
        "observation_dt": resolved["observation_dt"],
        "train_data_sha256": _batch_hash(train),
        "validation_data_sha256": _batch_hash(validation),
        "train_group_ids": sorted(train_groups),
        "validation_group_ids": sorted(validation_groups),
        "train_dataset_sources": train.metadata.get("dataset_sources", []),
        "validation_dataset_sources": validation.metadata.get("dataset_sources", []),
        "contact_class_weights": contact_weight_report,
        "horizon_schedule_seconds": [
            {"horizon": horizon, "seconds": horizon * resolved["observation_dt"],
             "epochs": epochs} for horizon, epochs in schedule
        ],
        "source_sha256": _source_hash(),
        "git_revision": _git_revision(),
        "python_version": __import__("platform").python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "device": resolved["device"],
    }
    normalizer = fit_normalizer(train, relative_features=resolved["relative_features"])
    model_config = DynamicsConfig(
        action_mode=resolved["action_mode"],
        relative_features=resolved["relative_features"],
        ensemble_size=resolved["ensemble_size"], hidden_dim=resolved["hidden_dim"],
        observation_dt=resolved["observation_dt"],
    )
    prior_state = None
    has_content = destination.exists() and any(destination.iterdir())
    if resume:
        if not has_content:
            raise ValueError("--resume requires an existing interrupted run")
        if resolved["device"] != "cpu":
            raise ValueError("exact training resume is supported on CPU only")
        try:
            old_config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
            old_manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            status = json.loads((destination / "status.json").read_text(encoding="utf-8"))
            if old_config != resolved:
                raise ValueError("resume config does not match the saved run")
            if any(old_manifest.get(key) != value for key, value in run_manifest.items()
                   if key != "git_revision"):
                raise ValueError("resume data, source, partition, or runtime manifest does not match")
            if status.get("status") == "complete":
                raise ValueError("run is already complete")
            loaded = DynamicsV3.load(destination / "last.pt", expected={
                "action_mode": resolved["action_mode"],
                "train_data_sha256": run_manifest["train_data_sha256"],
                "validation_data_sha256": run_manifest["validation_data_sha256"],
                "source_sha256": run_manifest["source_sha256"],
            })
            if loaded.normalizer is None or not np.array_equal(
                loaded.normalizer.state_mean, normalizer.state_mean
            ):
                raise ValueError("resume checkpoint normalizer does not match training data")
            raw_checkpoint = torch.load(destination / "last.pt", map_location="cpu", weights_only=False)
            prior_state = raw_checkpoint.get("training_state")
            if not isinstance(prior_state, dict):
                raise ValueError("resume checkpoint has no optimizer and RNG state")
            model = loaded
        except (OSError, json.JSONDecodeError, KeyError, RuntimeError) as error:
            raise ValueError(f"could not resume compatible run: {error}") from error
    else:
        if has_content:
            raise ValueError("output directory must be empty; pass --resume for a compatible interrupted run")
        destination.mkdir(parents=True, exist_ok=True)
        _atomic_json(destination / "config.json", resolved)
        _atomic_json(destination / "manifest.json", run_manifest)
        model = DynamicsV3(model_config, seed=resolved["seed"], normalizer=normalizer,
                           training_group_ids=tuple(train_groups),
                           validation_group_ids=tuple(validation_groups))
    model.training_group_ids = tuple(sorted(train_groups))
    model.validation_group_ids = tuple(sorted(validation_groups))
    model.training_metadata = run_manifest
    if prior_state is not None and prior_state.get("status") != "running":
        raise ValueError("run has no resumable epoch-boundary checkpoint")
    device = torch.device(resolved["device"])
    model.network.to(device)
    params = {name: torch.as_tensor(getattr(normalizer, name), dtype=torch.float32, device=device)
              for name in ("state_mean", "state_scale", "action_mean", "action_scale",
                           "delta_mean", "delta_scale")}
    params["contact_pos_weight"] = torch.as_tensor(
        contact_weight_values, dtype=torch.float32, device=device,
    )
    family_ids = np.asarray(sorted(train_groups))
    logs = destination / "training.jsonl"
    best_losses = [] if prior_state is None else list(prior_state["best_losses"])
    best_families = [] if prior_state is None else list(prior_state["best_families"])
    epochs_run = [] if prior_state is None else list(prior_state["epochs_run"])
    start_member = 0 if prior_state is None else int(prior_state["member_index"])
    sampler_history = [] if prior_state is None else list(prior_state.get("sampler_history", []))
    if prior_state is not None:
        model.network.load_state_dict(prior_state["network_state"], strict=True)

    def save_progress(member_index, next_epoch, optimizer_state, member_rng_state,
                      best_state, best_loss, best_breakdown, stale, stage_index,
                      sampler_history_value, member_sampling_value):
        state = {
            "status": "running", "member_index": member_index,
            "next_epoch": next_epoch, "optimizer_state": optimizer_state,
            "member_rng_state": member_rng_state, "best_state": best_state,
            "best_loss": best_loss, "best_breakdown": best_breakdown,
            "stale": stale, "stage_index": stage_index,
            "sampler_history": list(sampler_history_value),
            "member_sampling": list(member_sampling_value),
            "best_losses": list(best_losses),
            "best_families": list(best_families), "epochs_run": list(epochs_run),
            "network_state": {key: value.detach().cpu().clone()
                              for key, value in model.network.state_dict().items()},
            "torch_rng_state": torch.get_rng_state(),
        }
        model.network.eval()
        model.save(destination / "last.pt", run_manifest, training_state=state)
        _atomic_json(destination / "status.json", {
            "status": "running", "member_index": member_index,
            "next_epoch": next_epoch, "updated_unix": time.time(),
        })

    try:
        for member_index in range(start_member, len(model.network.members)):
            member = model.network.members[member_index]
            member_seed = resolved["seed"] + member_index * 997
            torch.manual_seed(member_seed)
            member_rng = np.random.default_rng(member_seed)
            bootstrap = member_rng.choice(family_ids, size=len(family_ids), replace=True)
            optimizer = torch.optim.AdamW(
                member.parameters(), lr=resolved["learning_rate"],
                weight_decay=resolved["weight_decay"],
            )
            best_state = None
            best_loss = float("inf")
            best_breakdown = {}
            stale = 0
            completed = 0
            epoch_start = 0
            active_stage_index = -1
            member_sampling = []
            if prior_state is not None and member_index == start_member:
                epoch_start = int(prior_state["next_epoch"])
                if prior_state["optimizer_state"] is not None:
                    member_rng.bit_generator.state = prior_state["member_rng_state"]
                    optimizer.load_state_dict(prior_state["optimizer_state"])
                    best_state = prior_state["best_state"]
                    best_loss = float(prior_state["best_loss"])
                    best_breakdown = prior_state["best_breakdown"]
                    stale = int(prior_state["stale"])
                    active_stage_index = int(prior_state.get("stage_index", -1))
                    sampler_history = list(prior_state.get("sampler_history", sampler_history))
                    member_sampling = list(prior_state.get("member_sampling", []))
                    completed = epoch_start
                    torch.set_rng_state(prior_state["torch_rng_state"])
                prior_state = None
            if epoch_start == 0:
                save_progress(member_index, 0, optimizer.state_dict(),
                              member_rng.bit_generator.state, best_state, best_loss,
                              best_breakdown, stale, active_stage_index, sampler_history,
                              member_sampling)
            for epoch in range(epoch_start, resolved["epochs"]):
                horizon = _horizon_for_epoch(schedule, epoch)
                stage_index = next(index for index, (stage_horizon, stage_epochs) in enumerate(schedule)
                                   if horizon == stage_horizon and epoch < sum(
                                       epochs for _, epochs in schedule[:index + 1]
                                   ))
                if stage_index != active_stage_index:
                    if best_state is not None:
                        member.load_state_dict(best_state)
                    active_stage_index = stage_index
                    best_state = None
                    best_loss = float("inf")
                    best_breakdown = {}
                    stale = 0
                started = time.perf_counter()
                member.train()
                selected_windows, sampling_diagnostics = _sample_training_windows(
                    train, horizon=horizon, stride=resolved["window_stride"],
                    count=len(train_pools[horizon]["uniform_windows"]), rng=member_rng,
                    event_fraction=resolved["event_fraction"], family_draws=bootstrap,
                    pools=train_pools[horizon],
                )
                member_sampling.append({"epoch": epoch + 1, "horizon": horizon,
                                        **sampling_diagnostics})
                permutation = member_rng.permutation(len(selected_windows))
                train_losses: dict[str, list[float]] = {}
                for offset in range(0, len(permutation), resolved["batch_size"]):
                    selected = [selected_windows[index]
                                for index in permutation[offset:offset + resolved["batch_size"]]]
                    indices = np.stack([window[2] for window in selected])
                    states = torch.as_tensor(train.states[indices], dtype=torch.float32, device=device)
                    actions = torch.as_tensor(train.actions[indices], dtype=torch.float32, device=device)
                    next_states = torch.as_tensor(train.next_states[indices], dtype=torch.float32,
                                                  device=device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, components = _window_loss(
                        member, states, actions, next_states, resolved, params,
                        train=True, return_components=True,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("training loss became non-finite")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(member.parameters(), resolved["gradient_clip"])
                    optimizer.step()
                    for name, value in components.items():
                        train_losses.setdefault(name, []).append(float(value.detach().cpu()))
                member.eval()
                validation_loss, family_losses = _validation_score(
                    member, validation, validation_windows[horizon], resolved, params,
                    resolved["batch_size"], device,
                )
                if not np.isfinite(validation_loss):
                    raise FloatingPointError("validation loss became non-finite")
                completed = epoch + 1
                if validation_loss < best_loss - 1e-6:
                    best_loss = validation_loss
                    best_breakdown = family_losses
                    best_state = {key: value.detach().clone()
                                  for key, value in member.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                with logs.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({
                        "member": member_index, "epoch": completed,
                        "horizon": horizon,
                        "horizon_seconds": float(horizon * resolved["observation_dt"]),
                        "train_loss": float(np.mean(train_losses["total"])),
                        "train_loss_components": {
                            name: float(np.mean(values)) for name, values in train_losses.items()
                        },
                        "sampling": sampling_diagnostics,
                        "validation_family_loss": validation_loss,
                        "validation_family_count": len(family_losses),
                        "seconds": time.perf_counter() - started,
                    }, allow_nan=False) + "\n")
                save_progress(member_index, completed, optimizer.state_dict(),
                              member_rng.bit_generator.state, best_state, best_loss,
                              best_breakdown, stale, active_stage_index, sampler_history,
                              member_sampling)
                stage_is_final = stage_index == len(schedule) - 1
                if stale >= resolved["patience"] and stage_is_final:
                    break
            if best_state is None:
                raise RuntimeError("training did not produce a finite validation checkpoint")
            member.load_state_dict(best_state)
            best_losses.append(best_loss)
            best_families.append(best_breakdown)
            epochs_run.append(completed)
            sampler_history.append({"member": member_index, "epochs": member_sampling})
            if member_index + 1 < len(model.network.members):
                save_progress(member_index + 1, 0, None, None, None, float("inf"), {}, 0,
                              -1, sampler_history, [])

        model.network.eval()
        model.save(destination / "best.pt", run_manifest)
        metrics = {
            "schema_version": 1,
            "model": asdict(model_config),
            "epochs_per_member": epochs_run,
            "best_validation_family_loss_per_member": best_losses,
            "best_validation_loss_by_family_per_member": best_families,
            "training_window_count_by_horizon": train_window_counts,
            "validation_window_count_by_horizon": validation_window_counts,
            "training_sampling_by_member": sampler_history,
            "contact_class_weights": contact_weight_report,
            "training_family_count": len(train_groups),
            "validation_family_count": len(validation_groups),
            "residual_gain": "all ones; calibration disabled",
            "run_manifest": run_manifest,
        }
        _atomic_json(destination / "metrics.json", metrics)
        _atomic_json(destination / "status.json", {
            "status": "complete", "finished_unix": time.time(),
            "epochs_per_member": epochs_run,
        })
    except BaseException as error:
        _atomic_json(destination / "status.json", {
            "status": "failed", "finished_unix": time.time(),
            "error_type": type(error).__name__, "error": str(error),
        })
        raise
    finally:
        model.network.to("cpu")
    return model
