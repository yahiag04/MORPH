"""Small action-conditioned predictor for one-step MuJoCo task dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # Keep the perception and MuJoCo environments PyTorch-free.
    torch = None
    nn = None

STATE_DIM = 19
ACTION_DIM = 4
CONTACT_STATE_INDICES = (17, 18)


def validate_transitions(
    states: np.ndarray, actions: np.ndarray, next_states: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return finite float32 one-step transitions."""
    try:
        state_array = np.asarray(states, dtype=np.float32)
        action_array = np.asarray(actions, dtype=np.float32)
        next_array = np.asarray(next_states, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("states, actions, and next_states must be numeric arrays") from error
    if state_array.ndim != 2 or state_array.shape[1] != STATE_DIM:
        raise ValueError(f"states must have shape (N, {STATE_DIM})")
    if action_array.shape != (state_array.shape[0], ACTION_DIM):
        raise ValueError(f"actions must have shape (N, {ACTION_DIM})")
    if next_array.shape != state_array.shape:
        raise ValueError(f"next_states must have shape (N, {STATE_DIM})")
    if state_array.shape[0] < 2:
        raise ValueError("at least two transitions are required")
    if not (np.isfinite(state_array).all() and np.isfinite(action_array).all()
            and np.isfinite(next_array).all()):
        raise ValueError("transition arrays must contain only finite values")
    return state_array.copy(), action_array.copy(), next_array.copy()


@dataclass(frozen=True)
class ClipSplit:
    """Deterministic row masks and clip identifiers for train/validation."""

    train_mask: np.ndarray
    validation_mask: np.ndarray
    train_clip_ids: tuple[str, ...]
    validation_clip_ids: tuple[str, ...]


def split_by_clip(
    clip_ids: np.ndarray, *, validation_fraction: float = 0.25, seed: int = 17
) -> ClipSplit:
    """Split whole clips before training so adjacent states cannot leak."""
    clips = np.asarray(clip_ids).astype(str)
    if clips.ndim != 1 or clips.size < 2 or np.any(clips == ""):
        raise ValueError("clip_ids must contain at least two non-empty clip identifiers")
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    unique = np.unique(clips)
    if unique.size < 2:
        raise ValueError("at least two unique clips are required for validation")
    validation_count = min(unique.size - 1, max(1, int(np.ceil(unique.size * validation_fraction))))
    rng = np.random.default_rng(seed)
    validation = set(rng.choice(unique, size=validation_count, replace=False).tolist())
    validation_mask = np.fromiter((clip in validation for clip in clips), dtype=bool, count=len(clips))
    train_mask = ~validation_mask
    return ClipSplit(
        train_mask=train_mask, validation_mask=validation_mask,
        train_clip_ids=tuple(sorted(set(clips[train_mask]))),
        validation_clip_ids=tuple(sorted(validation)),
    )


@dataclass(frozen=True)
class DynamicsNormalizer:
    """Training-only means and scales for model input and state delta."""

    state_mean: np.ndarray
    state_scale: np.ndarray
    action_mean: np.ndarray
    action_scale: np.ndarray
    delta_mean: np.ndarray
    delta_scale: np.ndarray

    @classmethod
    def fit(
        cls, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray
    ) -> "DynamicsNormalizer":
        state_mean = states.mean(axis=0)
        state_scale = states.std(axis=0)
        action_mean = actions.mean(axis=0)
        action_scale = actions.std(axis=0)
        delta = next_states - states
        delta_mean = delta.mean(axis=0)
        delta_scale = delta.std(axis=0)
        for index in CONTACT_STATE_INDICES:
            state_mean[index] = 0.0
            state_scale[index] = 1.0
            delta_mean[index] = 0.0
            delta_scale[index] = 1.0
        state_scale = np.maximum(state_scale, 1e-5)
        action_scale = np.maximum(action_scale, 1e-5)
        delta_scale = np.maximum(delta_scale, 1e-5)
        return cls(*(np.asarray(value, dtype=np.float32) for value in (
            state_mean, state_scale, action_mean, action_scale, delta_mean, delta_scale
        )))


if nn is not None:
    class StateDynamicsModel(nn.Module):
        """19-state, 4-action MLP that predicts a normalized state delta."""

        def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                     hidden_dim: int = 128) -> None:
            super().__init__()
            if state_dim != STATE_DIM or action_dim != ACTION_DIM or hidden_dim < 1:
                raise ValueError("model dimensions must be state=19, action=4, hidden>=1")
            self.network = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, state_dim),
            )

        def forward(self, normalized_state, normalized_action):
            return self.network(torch.cat((normalized_state, normalized_action), dim=-1))
else:
    class StateDynamicsModel:  # type: ignore[no-redef]
        """Placeholder that gives a clear message in the lightweight runtime."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("StateDynamicsModel requires the optional PyTorch training runtime")


@dataclass
class StateDynamicsFit:
    """Trained predictor and held-out aggregate/per-clip metrics."""

    model: Any
    normalizer: DynamicsNormalizer
    train_clip_ids: tuple[str, ...]
    validation_clip_ids: tuple[str, ...]
    metrics: dict


def _predict_next(model, normalizer: DynamicsNormalizer, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    if torch is None:
        raise ImportError("world-model training requires the optional PyTorch runtime")
    model.eval()
    with torch.no_grad():
        normalized_state = (states - normalizer.state_mean) / normalizer.state_scale
        normalized_action = (actions - normalizer.action_mean) / normalizer.action_scale
        predicted_delta = model(
            torch.as_tensor(normalized_state, dtype=torch.float32),
            torch.as_tensor(normalized_action, dtype=torch.float32),
        ).cpu().numpy()
    return states + normalizer.delta_mean + predicted_delta * normalizer.delta_scale


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error), dtype=np.float64)))


def _rollout_metrics(
    model, normalizer: DynamicsNormalizer, states: np.ndarray, actions: np.ndarray,
    next_states: np.ndarray, clip_ids: np.ndarray, episode_ids: np.ndarray,
    validation_mask: np.ndarray, horizon: int,
) -> tuple[dict, dict]:
    by_clip: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for episode in np.unique(episode_ids[validation_mask]):
        indices = np.flatnonzero((episode_ids == episode) & validation_mask)
        indices.sort()
        if len(indices) < horizon:
            continue
        clip = str(clip_ids[indices[0]])
        for start in range(0, len(indices) - horizon + 1, horizon):
            window = indices[start : start + horizon]
            prediction = states[window[0]].copy()
            for index in window:
                prediction = _predict_next(
                    model, normalizer, prediction.reshape(1, -1), actions[index].reshape(1, -1)
                )[0]
            truth = next_states[window[-1]]
            persistence = states[window[0]]
            by_clip.setdefault(clip, []).append((prediction - truth, persistence - truth))
    clip_metrics: dict[str, dict] = {}
    learned_errors, persistence_errors = [], []
    for clip, pairs in sorted(by_clip.items()):
        learned = np.concatenate([pair[0].reshape(1, -1) for pair in pairs], axis=0)
        persisted = np.concatenate([pair[1].reshape(1, -1) for pair in pairs], axis=0)
        clip_metrics[clip] = {
            "world_model_rmse": _rmse(learned),
            "persistence_rmse": _rmse(persisted),
            "rollouts": int(len(pairs)),
        }
        learned_errors.append(learned)
        persistence_errors.append(persisted)
    if not learned_errors:
        return {"world_model_rmse": None, "persistence_rmse": None, "rollouts": 0}, clip_metrics
    return {
        "world_model_rmse": _rmse(np.concatenate(learned_errors)),
        "persistence_rmse": _rmse(np.concatenate(persistence_errors)),
        "rollouts": int(sum(len(value) for value in by_clip.values())),
    }, clip_metrics


def fit_state_dynamics(
    states: np.ndarray, actions: np.ndarray, next_states: np.ndarray,
    clip_ids: np.ndarray, *, episode_ids: np.ndarray | None = None,
    validation_fraction: float = 0.25, seed: int = 17, epochs: int = 400,
    batch_size: int = 256, learning_rate: float = 1e-3,
    rollout_horizon: int = 25, patience: int = 40,
) -> StateDynamicsFit:
    """Train an MLP using whole training clips and report held-out errors."""
    if torch is None:
        raise ImportError("world-model training requires the optional PyTorch runtime")
    states, actions, next_states = validate_transitions(states, actions, next_states)
    clip_ids = np.asarray(clip_ids).astype(str)
    if clip_ids.shape != (len(states),):
        raise ValueError("clip_ids must have one identifier per transition")
    if episode_ids is None:
        episode_ids = clip_ids.copy()
    episode_ids = np.asarray(episode_ids).astype(str)
    if episode_ids.shape != clip_ids.shape:
        raise ValueError("episode_ids must have one identifier per transition")
    if epochs < 1 or batch_size < 1 or rollout_horizon < 1 or patience < 1:
        raise ValueError("epochs, batch_size, rollout_horizon, and patience must be positive")

    split = split_by_clip(clip_ids, validation_fraction=validation_fraction, seed=seed)
    normalizer = DynamicsNormalizer.fit(states[split.train_mask], actions[split.train_mask], next_states[split.train_mask])
    state_norm = (states - normalizer.state_mean) / normalizer.state_scale
    action_norm = (actions - normalizer.action_mean) / normalizer.action_scale
    delta = next_states - states
    delta_norm = (delta - normalizer.delta_mean) / normalizer.delta_scale
    train_indices = np.flatnonzero(split.train_mask)
    val_indices = np.flatnonzero(split.validation_mask)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = StateDynamicsModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    state_tensor = torch.as_tensor(state_norm, dtype=torch.float32)
    action_tensor = torch.as_tensor(action_norm, dtype=torch.float32)
    delta_tensor = torch.as_tensor(delta_norm, dtype=torch.float32)
    best_state = None
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    epochs_run = 0
    for epoch in range(epochs):
        model.train()
        shuffled = rng.permutation(train_indices)
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(state_tensor[indices], action_tensor[indices])
            loss = loss_fn(prediction, delta_tensor[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_prediction = model(state_tensor[val_indices], action_tensor[val_indices])
            validation_loss = float(loss_fn(validation_prediction, delta_tensor[val_indices]).item())
        epochs_run = epoch + 1
        if validation_loss < best_validation_loss - 1e-7:
            best_validation_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    prediction = _predict_next(model, normalizer, states[val_indices], actions[val_indices])
    one_step_error = prediction - next_states[val_indices]
    persistence_error = states[val_indices] - next_states[val_indices]
    per_clip: dict[str, dict] = {}
    for clip in split.validation_clip_ids:
        indices = val_indices[clip_ids[val_indices] == clip]
        per_clip[clip] = {
            "one_step_world_model_rmse": _rmse(prediction[np.isin(val_indices, indices)] - next_states[indices]),
            "one_step_persistence_rmse": _rmse(states[indices] - next_states[indices]),
            "transitions": int(len(indices)),
        }
    rollout, rollout_by_clip = _rollout_metrics(
        model, normalizer, states, actions, next_states, clip_ids, episode_ids,
        split.validation_mask, rollout_horizon,
    )
    for clip, metrics in per_clip.items():
        if clip in rollout_by_clip:
            metrics.update(rollout_by_clip[clip])
    metrics = {
        "state_dim": STATE_DIM, "action_dim": ACTION_DIM, "hidden_dim": 128,
        "train_clip_count": len(split.train_clip_ids),
        "validation_clip_count": len(split.validation_clip_ids),
        "validation_transition_count": int(len(val_indices)),
        "epochs_run": epochs_run,
        "one_step_rmse": _rmse(one_step_error),
        "persistence_one_step_rmse": _rmse(persistence_error),
        "rollout_horizon_steps": int(rollout_horizon),
        "rollout_seconds": float(rollout_horizon * 0.02),
        "rollout_rmse": rollout["world_model_rmse"],
        "persistence_rollout_rmse": rollout["persistence_rmse"],
        "rollout_count": rollout["rollouts"],
        "per_clip": per_clip,
    }
    return StateDynamicsFit(model, normalizer, split.train_clip_ids, split.validation_clip_ids, metrics)


def save_checkpoint(fit: StateDynamicsFit, destination: str) -> None:
    """Save the model and normalizer for local prediction only."""
    if torch is None:
        raise ImportError("saving world-model checkpoints requires PyTorch")
    torch.save({
        "state_dim": STATE_DIM, "action_dim": ACTION_DIM, "hidden_dim": 128,
        "model_state_dict": fit.model.state_dict(),
        "normalizer": {name: getattr(fit.normalizer, name) for name in (
            "state_mean", "state_scale", "action_mean", "action_scale", "delta_mean", "delta_scale"
        )},
        "train_clip_ids": fit.train_clip_ids,
        "validation_clip_ids": fit.validation_clip_ids,
    }, destination)
