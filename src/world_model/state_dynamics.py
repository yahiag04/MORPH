"""Ensemble dynamics model for the MuJoCo contact manipulation task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # Keep the perception and MuJoCo environments PyTorch-free.
    torch = None
    nn = None
    F = None

STATE_DIM = 37
ACTION_DIM = 4
CONTACT_STATE_INDICES = (32, 33)
QUATERNION_STATE_SLICE = slice(20, 24)
CONTINUOUS_STATE_INDICES = tuple(i for i in range(STATE_DIM) if i not in CONTACT_STATE_INDICES)
DYNAMIC_STATE_INDICES = tuple(range(32))
STATE_GROUPS = {
    "ee_position": slice(0, 3),
    "arm_position": slice(3, 10),
    "arm_velocity": slice(10, 17),
    "package_position": slice(17, 20),
    "package_orientation": QUATERNION_STATE_SLICE,
    "package_linear_velocity": slice(24, 27),
    "package_angular_velocity": slice(27, 30),
    "gripper_aperture": slice(30, 31),
    "gripper_velocity": slice(31, 32),
    "tray_center": slice(34, 37),
}


def validate_transitions(
    states: np.ndarray, actions: np.ndarray, next_states: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate finite transitions and require unit quaternions and binary contacts."""
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
    for name, values in (("states", state_array), ("next_states", next_array)):
        quaternion = values[:, QUATERNION_STATE_SLICE]
        if not np.allclose(np.linalg.norm(quaternion, axis=1), 1.0, atol=1e-3):
            raise ValueError(f"{name} package quaternions must have unit norm")
        contacts = values[:, CONTACT_STATE_INDICES]
        if not np.isin(contacts, (0.0, 1.0)).all():
            raise ValueError(f"{name} contact flags must be binary")
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
    """Training-only means and scales for continuous state, action, and deltas."""

    state_mean: np.ndarray
    state_scale: np.ndarray
    action_mean: np.ndarray
    action_scale: np.ndarray
    delta_mean: np.ndarray
    delta_scale: np.ndarray

    @classmethod
    def fit(cls, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> "DynamicsNormalizer":
        state_mean = states.mean(axis=0)
        state_scale = states.std(axis=0)
        action_mean = actions.mean(axis=0)
        action_scale = actions.std(axis=0)
        delta = next_states - states
        delta_mean = delta.mean(axis=0)
        delta_scale = delta.std(axis=0)
        for index in CONTACT_STATE_INDICES:
            state_mean[index] = delta_mean[index] = 0.0
            state_scale[index] = delta_scale[index] = 1.0
        delta_scale[QUATERNION_STATE_SLICE] = np.maximum(
            delta_scale[QUATERNION_STATE_SLICE], 0.01
        )
        # Avoid amplifying sensor quantization and make stationary dimensions
        # numerically well behaved during autoregressive rollouts.
        state_scale[:32] = np.maximum(state_scale[:32], 0.02)
        state_scale[34:37] = np.maximum(state_scale[34:37], 0.02)
        action_scale = np.maximum(action_scale, 1e-5)
        delta_scale[:32] = np.maximum(delta_scale[:32], 0.01)
        delta_mean[34:37] = 0.0
        delta_scale[34:37] = 1.0
        return cls(*(np.asarray(value, dtype=np.float32) for value in (
            state_mean, state_scale, action_mean, action_scale, delta_mean, delta_scale
        )))


if nn is not None:
    class StateDynamicsModel(nn.Module):
        """One probabilistic-ready ensemble member with separate contact logits."""

        def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                     hidden_dim: int = 128) -> None:
            super().__init__()
            if state_dim != STATE_DIM or action_dim != ACTION_DIM or hidden_dim < 1:
                raise ValueError("model dimensions must be state=37, action=4, hidden>=1")
            self.trunk = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            )
            self.continuous_head = nn.Linear(hidden_dim, len(DYNAMIC_STATE_INDICES))
            self.contact_head = nn.Linear(hidden_dim, len(CONTACT_STATE_INDICES))

        def forward(self, normalized_state, normalized_action):
            features = self.trunk(torch.cat((normalized_state, normalized_action), dim=-1))
            return self.continuous_head(features), self.contact_head(features)


    class StateDynamicsEnsemble(nn.Module):
        """Bootstrap ensemble; each member receives whole-clip resamples."""

        def __init__(self, count: int = 5, hidden_dim: int = 128) -> None:
            super().__init__()
            if count < 1:
                raise ValueError("ensemble count must be positive")
            self.members = nn.ModuleList(StateDynamicsModel(hidden_dim=hidden_dim) for _ in range(count))
            self.register_buffer("residual_gain", torch.ones(len(DYNAMIC_STATE_INDICES)))

        def forward(self, normalized_state, normalized_action):
            outputs = [member(normalized_state, normalized_action) for member in self.members]
            continuous = torch.stack([item[0] for item in outputs])
            contacts = torch.stack([item[1] for item in outputs])
            return continuous.mean(dim=0), contacts.mean(dim=0), continuous, contacts
else:
    class StateDynamicsModel:  # type: ignore[no-redef]
        """Placeholder that gives a clear message in the lightweight runtime."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("StateDynamicsModel requires the optional PyTorch training runtime")


    class StateDynamicsEnsemble:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("StateDynamicsEnsemble requires the optional PyTorch training runtime")


@dataclass
class StateDynamicsFit:
    """Trained predictor, normalizer, clip partitions, and held-out metrics."""

    model: Any
    normalizer: DynamicsNormalizer
    train_clip_ids: tuple[str, ...]
    validation_clip_ids: tuple[str, ...]
    metrics: dict


def _normalizer_tensors(normalizer: DynamicsNormalizer, device="cpu") -> dict[str, Any]:
    return {name: torch.as_tensor(getattr(normalizer, name), dtype=torch.float32, device=device)
            for name in ("state_mean", "state_scale", "action_mean", "action_scale", "delta_mean", "delta_scale")}


def _step_torch(model, state, action, normalizer, *, hard_contacts: bool = False):
    params = _normalizer_tensors(normalizer, state.device) if isinstance(normalizer, DynamicsNormalizer) else normalizer
    normalized_state = (state - params["state_mean"]) / params["state_scale"]
    normalized_action = (action - params["action_mean"]) / params["action_scale"]
    output = model(normalized_state, normalized_action)
    if len(output) == 4:
        continuous_delta, contact_logits = output[:2]
    else:
        continuous_delta, contact_logits = output
    indices = torch.as_tensor(DYNAMIC_STATE_INDICES, device=state.device)
    residual_gain = getattr(model, "residual_gain", 1.0)
    delta = params["delta_mean"][indices] + continuous_delta * params["delta_scale"][indices]
    next_dynamic = state[:, indices] + residual_gain * delta
    next_dynamic = torch.cat((next_dynamic[:, :20],
                              F.normalize(next_dynamic[:, 20:24], p=2, dim=-1, eps=1e-8),
                              next_dynamic[:, 24:]), dim=-1)
    contact_probabilities = torch.sigmoid(contact_logits)
    if hard_contacts:
        contact_values = (contact_probabilities >= 0.5).to(state.dtype)
    else:
        contact_values = contact_probabilities
    contacts = {index: offset for offset, index in enumerate(CONTACT_STATE_INDICES)}
    next_state = torch.cat((next_dynamic, contact_values, state[:, 34:37]), dim=-1)
    return next_state, contact_logits, contact_probabilities


def predict_next_state(
    model: StateDynamicsEnsemble, normalizer: DynamicsNormalizer,
    states: np.ndarray, actions: np.ndarray, *, hard_contacts: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ensemble mean next states, contact probabilities, and normalized spread."""
    if torch is None:
        raise ImportError("world-model prediction requires the optional PyTorch runtime")
    state_array, action_array, _ = validate_transitions(
        states, actions, np.asarray(states, dtype=np.float32)
    )
    model.eval()
    with torch.no_grad():
        state_tensor = torch.as_tensor(state_array)
        action_tensor = torch.as_tensor(action_array)
        next_state, _, contact_probabilities = _step_torch(
            model, state_tensor, action_tensor, normalizer, hard_contacts=hard_contacts
        )
        params = _normalizer_tensors(normalizer)
        normalized_state = (state_tensor - params["state_mean"]) / params["state_scale"]
        normalized_action = (action_tensor - params["action_mean"]) / params["action_scale"]
        member_outputs = [member(normalized_state, normalized_action) for member in model.members]
        member_continuous = torch.stack([item[0] for item in member_outputs])
        spread = member_continuous.std(dim=0, unbiased=False).mean(dim=-1)
    return next_state.numpy(), contact_probabilities.numpy(), spread.numpy()


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error), dtype=np.float64)))


def _make_windows(episode_ids: np.ndarray, mask: np.ndarray, horizon: int, stride: int = 5) -> list[np.ndarray]:
    windows = []
    for episode in np.unique(episode_ids[mask]):
        indices = np.flatnonzero((episode_ids == episode) & mask)
        if len(indices) < horizon:
            continue
        for start in range(0, len(indices) - horizon + 1, stride):
            windows.append(indices[start:start + horizon])
    return windows


def _continuous_loss(predicted, target, scale):
    indices = torch.as_tensor(DYNAMIC_STATE_INDICES, device=predicted.device)
    return F.mse_loss((predicted[:, indices] - target[:, indices]) / scale[indices], torch.zeros_like(predicted[:, indices]))


def _window_objective(member, states, actions, next_states, normalizer, windows, *, train: bool):
    params = _normalizer_tensors(normalizer, states.device)
    index = torch.as_tensor(np.asarray(windows), dtype=torch.long, device=states.device)
    predicted = states[index[:, 0]]
    one_step_losses = []
    horizon_losses = []
    contact_indices = torch.as_tensor(CONTACT_STATE_INDICES, device=states.device)
    for step in range(index.shape[1]):
        rows = index[:, step]
        truth = next_states[rows]
        cont_delta, contact_logits = member(
            (predicted - params["state_mean"]) / params["state_scale"],
            (actions[rows] - params["action_mean"]) / params["action_scale"],
        )
        cont_indices = torch.as_tensor(DYNAMIC_STATE_INDICES, device=states.device)
        target_delta = (truth[:, cont_indices] - predicted[:, cont_indices] - params["delta_mean"][cont_indices]) / params["delta_scale"][cont_indices]
        one_step_losses.append(F.mse_loss(cont_delta, target_delta) + F.binary_cross_entropy_with_logits(contact_logits, truth[:, contact_indices]))
        predicted, _, _ = _step_torch(member, predicted, actions[rows], params)
        if step + 1 in (1, 5, 25):
            state_loss = _continuous_loss(predicted, truth, params["state_scale"])
            contact_loss = F.binary_cross_entropy(predicted[:, contact_indices].clamp(1e-6, 1 - 1e-6), truth[:, contact_indices])
            horizon_losses.append(state_loss + contact_loss)
    one_step_loss = torch.stack(one_step_losses).mean()
    rollout_loss = torch.stack(horizon_losses).mean()
    return 0.5 * one_step_loss + 0.5 * rollout_loss


def _group_metrics(predicted, truth, persistence, normalizer, contact_probs, contact_truth) -> dict:
    metrics = {}
    for name, feature_slice in STATE_GROUPS.items():
        p_error = predicted[:, feature_slice] - truth[:, feature_slice]
        b_error = persistence[:, feature_slice] - truth[:, feature_slice]
        metrics[name] = {"one_step_rmse": _rmse(p_error), "persistence_one_step_rmse": _rmse(b_error)}
    for name, feature_slice in STATE_GROUPS.items():
        indices = np.arange(STATE_DIM)[feature_slice]
        scale = normalizer.state_scale[indices]
        metrics[name]["normalized_one_step_rmse"] = _rmse((predicted[:, feature_slice] - truth[:, feature_slice]) / scale)
    predicted_contacts = (contact_probs >= 0.5).astype(np.float32)
    tp = int(((predicted_contacts == 1) & (contact_truth == 1)).sum())
    fp = int(((predicted_contacts == 1) & (contact_truth == 0)).sum())
    fn = int(((predicted_contacts == 0) & (contact_truth == 1)).sum())
    metrics["contact_f1"] = 2 * tp / max(1, 2 * tp + fp + fn)
    metrics["contact_accuracy"] = float(np.mean(predicted_contacts == contact_truth))
    return metrics


def _evaluate_rollouts(model, states, actions, next_states, clips, episodes, mask,
                       normalizer, horizon):
    grouped_errors: dict[str, dict[str, list[np.ndarray]]] = {}
    contact_predictions = []
    contact_truths = []
    disagreement = []
    params = _normalizer_tensors(normalizer)
    model.eval()
    with torch.no_grad():
        for episode in np.unique(episodes[mask]):
            indices = np.flatnonzero((episodes == episode) & mask)
            for start in range(0, len(indices) - horizon + 1, horizon):
                window = indices[start:start + horizon]
                clip = str(clips[window[0]])
                predicted = torch.as_tensor(states[window[0]:window[0] + 1])
                for index in window:
                    action = torch.as_tensor(actions[index:index + 1])
                    predicted, _, probs = _step_torch(model, predicted, action, params, hard_contacts=True)
                truth = next_states[window[-1]:window[-1] + 1]
                persist = states[window[0]:window[0] + 1]
                entry = grouped_errors.setdefault(clip, {"model": [], "persistence": []})
                entry["model"].append(predicted.numpy()[0] - truth[0])
                entry["persistence"].append(persist[0] - truth[0])
                contact_predictions.append((probs.numpy()[0] >= 0.5).astype(np.float32))
                contact_truths.append(truth[0, CONTACT_STATE_INDICES])
                disagreement.append(float(_ensemble_disagreement(model, states[window[0]], actions[window[0]], params)))
    output = {name: {"world_model_rmse": None, "persistence_rmse": None} for name in STATE_GROUPS}
    if not grouped_errors:
        return output, {"contact_accuracy": None, "contact_f1": None, "rollouts": 0}, None
    for name, feature_slice in STATE_GROUPS.items():
        learned = np.concatenate([np.stack(rows["model"])[:, feature_slice] for rows in grouped_errors.values()])
        persisted = np.concatenate([np.stack(rows["persistence"])[:, feature_slice] for rows in grouped_errors.values()])
        output[name] = {"world_model_rmse": _rmse(learned), "persistence_rmse": _rmse(persisted)}
    contacts_pred = np.stack(contact_predictions)
    contacts_true = np.stack(contact_truths)
    tp = int(((contacts_pred == 1) & (contacts_true == 1)).sum())
    fp = int(((contacts_pred == 1) & (contacts_true == 0)).sum())
    fn = int(((contacts_pred == 0) & (contacts_true == 1)).sum())
    contact_metrics = {"contact_accuracy": float(np.mean(contacts_pred == contacts_true)),
                       "contact_f1": 2 * tp / max(1, 2 * tp + fp + fn),
                       "rollouts": len(contacts_pred)}
    return output, contact_metrics, float(np.mean(disagreement))


def _ensemble_disagreement(model, state, action, params) -> float:
    state_t = torch.as_tensor(np.asarray(state).reshape(1, -1), dtype=torch.float32)
    action_t = torch.as_tensor(np.asarray(action).reshape(1, -1), dtype=torch.float32)
    normalized_state = (state_t - params["state_mean"]) / params["state_scale"]
    normalized_action = (action_t - params["action_mean"]) / params["action_scale"]
    outputs = [member(normalized_state, normalized_action)[0] for member in model.members]
    return float(torch.stack(outputs).std(dim=0, unbiased=False).mean().item())


def calibrate_rollout_gains(
    model, normalizer: DynamicsNormalizer, states: np.ndarray, actions: np.ndarray,
    next_states: np.ndarray, clip_ids: np.ndarray, episode_ids: np.ndarray,
    *, horizon: int = 25,
) -> dict[str, float]:
    """Tune group residual gains on validation episodes to limit rollout drift."""
    candidates = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    gain = np.ones(len(DYNAMIC_STATE_INDICES), dtype=np.float32)
    model.residual_gain.copy_(torch.as_tensor(gain))
    clips = np.asarray(clip_ids).astype(str)
    episodes = np.asarray(episode_ids).astype(str)
    mask = np.ones(len(states), dtype=bool)
    params = _normalizer_tensors(normalizer)
    groups = {name: feature_slice for name, feature_slice in STATE_GROUPS.items()
              if name != "tray_center"}
    for _ in range(2):
        for name, feature_slice in groups.items():
            indices = np.arange(STATE_DIM)[feature_slice]
            indices = indices[indices < len(DYNAMIC_STATE_INDICES)]
            if not len(indices):
                continue
            best_gain = float(gain[indices[0]])
            best_error = float("inf")
            for candidate in candidates:
                gain[indices] = candidate
                model.residual_gain.copy_(torch.as_tensor(gain))
                rollout, _, _ = _evaluate_rollouts(
                    model, states, actions, next_states, clips, episodes,
                    mask, normalizer, horizon,
                )
                error = rollout[name]["world_model_rmse"]
                if error is not None and error < best_error:
                    best_error, best_gain = error, candidate
            gain[indices] = best_gain
            model.residual_gain.copy_(torch.as_tensor(gain))
    return {name: float(np.mean(gain[np.arange(STATE_DIM)[feature_slice]
                                      [np.arange(STATE_DIM)[feature_slice] < 32]]))
            for name, feature_slice in groups.items()
            if np.any(np.arange(STATE_DIM)[feature_slice] < 32)}


def fit_state_dynamics(
    states: np.ndarray, actions: np.ndarray, next_states: np.ndarray,
    clip_ids: np.ndarray, *, episode_ids: np.ndarray | None = None,
    validation_fraction: float = 0.25, seed: int = 17, epochs: int = 160,
    batch_size: int = 16, learning_rate: float = 5e-4,
    rollout_horizon: int = 25, patience: int = 20, ensemble_size: int = 5,
    transition_dt_seconds: float | None = None,
) -> StateDynamicsFit:
    """Fit clip-bootstrapped dynamics using one-step and autoregressive losses."""
    if torch is None:
        raise ImportError("world-model training requires the optional PyTorch runtime")
    if transition_dt_seconds is not None and (not np.isfinite(transition_dt_seconds) or transition_dt_seconds <= 0):
        raise ValueError("transition_dt_seconds must be finite and positive")
    states, actions, next_states = validate_transitions(states, actions, next_states)
    clip_ids = np.asarray(clip_ids).astype(str)
    if clip_ids.shape != (len(states),):
        raise ValueError("clip_ids must have one identifier per transition")
    episode_ids = clip_ids.copy() if episode_ids is None else np.asarray(episode_ids).astype(str)
    if episode_ids.shape != clip_ids.shape:
        raise ValueError("episode_ids must have one identifier per transition")
    if epochs < 1 or batch_size < 1 or rollout_horizon < 25 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive; rollout_horizon must be >=25")

    split = split_by_clip(clip_ids, validation_fraction=validation_fraction, seed=seed)
    normalizer = DynamicsNormalizer.fit(states[split.train_mask], actions[split.train_mask], next_states[split.train_mask])
    train_windows = _make_windows(episode_ids, split.train_mask, rollout_horizon)
    val_windows = _make_windows(episode_ids, split.validation_mask, rollout_horizon)
    if not train_windows or not val_windows:
        raise ValueError("training and validation episodes must each contain a full rollout window")
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    params = _normalizer_tensors(normalizer)
    state_tensor = torch.as_tensor(states, dtype=torch.float32)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32)
    next_tensor = torch.as_tensor(next_states, dtype=torch.float32)
    ensemble = StateDynamicsEnsemble(ensemble_size)
    epochs_run = []
    best_validation_losses = []

    clip_windows = {clip: [] for clip in split.train_clip_ids}
    for window in train_windows:
        clip_windows[str(clip_ids[window[0]])].append(window)
    validation_tensor = torch.as_tensor(np.asarray(val_windows), dtype=torch.long)

    for member_index, member in enumerate(ensemble.members):
        member_seed = seed + member_index * 997
        torch.manual_seed(member_seed)
        rng = np.random.default_rng(member_seed)
        sampled_clips = rng.choice(split.train_clip_ids, size=len(split.train_clip_ids), replace=True)
        bootstrap_windows = [window for clip in sampled_clips for window in clip_windows[clip]]
        optimizer = torch.optim.AdamW(member.parameters(), lr=learning_rate, weight_decay=1e-5)
        best_state, best_loss = None, float("inf")
        stale_epochs = 0
        completed = 0
        for epoch in range(epochs):
            member.train()
            permutation = rng.permutation(len(bootstrap_windows))
            for start in range(0, len(permutation), batch_size):
                batch = [bootstrap_windows[i] for i in permutation[start:start + batch_size]]
                optimizer.zero_grad(set_to_none=True)
                loss = _window_objective(member, state_tensor, action_tensor, next_tensor,
                                         normalizer, batch, train=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(member.parameters(), max_norm=5.0)
                optimizer.step()
            member.eval()
            with torch.no_grad():
                val_losses = []
                for start in range(0, len(validation_tensor), batch_size):
                    batch = validation_tensor[start:start + batch_size].cpu().numpy()
                    val_losses.append(_window_objective(member, state_tensor, action_tensor,
                                                        next_tensor, normalizer, batch, train=False).item())
                validation_loss = float(np.mean(val_losses))
            completed = epoch + 1
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_state = {key: value.detach().clone() for key, value in member.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break
        if best_state is not None:
            member.load_state_dict(best_state)
        epochs_run.append(completed)
        best_validation_losses.append(best_loss)

    validation_indices = np.flatnonzero(split.validation_mask)
    rollout_gains = calibrate_rollout_gains(
        ensemble, normalizer, states[validation_indices], actions[validation_indices],
        next_states[validation_indices], clip_ids[validation_indices],
        episode_ids[validation_indices], horizon=rollout_horizon,
    )

    val_indices = validation_indices
    predicted, contact_probabilities, spread = predict_next_state(
        ensemble, normalizer, states[val_indices], actions[val_indices], hard_contacts=False
    )
    groups = _group_metrics(predicted, next_states[val_indices], states[val_indices], normalizer,
                            contact_probabilities, next_states[val_indices][:, CONTACT_STATE_INDICES])
    rollout_groups, rollout_contacts, disagreement = _evaluate_rollouts(
        ensemble, states, actions, next_states, clip_ids, episode_ids, split.validation_mask,
        normalizer, rollout_horizon,
    )
    for name in STATE_GROUPS:
        groups[name].update(rollout_groups[name])
    normalized_error = (predicted[:, CONTINUOUS_STATE_INDICES] - next_states[val_indices][:, CONTINUOUS_STATE_INDICES]) / normalizer.state_scale[list(CONTINUOUS_STATE_INDICES)]
    persistence_normalized_error = (states[val_indices][:, CONTINUOUS_STATE_INDICES] - next_states[val_indices][:, CONTINUOUS_STATE_INDICES]) / normalizer.state_scale[list(CONTINUOUS_STATE_INDICES)]
    metrics = {
        "state_dim": STATE_DIM, "action_dim": ACTION_DIM, "hidden_dim": 128,
        "ensemble_size": ensemble_size, "train_clip_count": len(split.train_clip_ids),
        "validation_clip_count": len(split.validation_clip_ids),
        "validation_transition_count": int(len(val_indices)),
        "epochs_per_member": epochs_run, "best_validation_loss_per_member": best_validation_losses,
        "one_step_normalized_rmse": _rmse(normalized_error),
        "persistence_one_step_normalized_rmse": _rmse(persistence_normalized_error),
        "rollout_horizon_steps": int(rollout_horizon),
        "transition_dt_seconds": transition_dt_seconds,
        "rollout_seconds": None if transition_dt_seconds is None else float(rollout_horizon * transition_dt_seconds),
        "rollout_position_group_rmse": _rmse(np.asarray([groups[name]["world_model_rmse"] for name in ("ee_position", "arm_position", "package_position")])),
        "persistence_rollout_position_group_rmse": _rmse(np.asarray([groups[name]["persistence_rmse"] for name in ("ee_position", "arm_position", "package_position")])),
        "rollout_count": rollout_contacts["rollouts"],
        "contact_accuracy": groups["contact_accuracy"], "contact_f1": groups["contact_f1"],
        "rollout_contact_accuracy": rollout_contacts["contact_accuracy"],
        "rollout_contact_f1": rollout_contacts["contact_f1"],
        "ensemble_disagreement_normalized": disagreement,
        "ensemble_disagreement_one_step_normalized": float(np.mean(spread)),
        "validation_calibrated_residual_gains": rollout_gains,
        "groups": groups,
    }
    return StateDynamicsFit(ensemble, normalizer, split.train_clip_ids, split.validation_clip_ids, metrics)


def evaluate_state_dynamics(
    fit: StateDynamicsFit, states: np.ndarray, actions: np.ndarray,
    next_states: np.ndarray, clip_ids: np.ndarray, *,
    episode_ids: np.ndarray | None = None, rollout_horizon: int = 25,
    transition_dt_seconds: float | None = None,
) -> dict:
    """Evaluate a fitted model on a fully separate set of clip groups."""
    states, actions, next_states = validate_transitions(states, actions, next_states)
    clips = np.asarray(clip_ids).astype(str)
    episodes = clips.copy() if episode_ids is None else np.asarray(episode_ids).astype(str)
    if clips.shape != (len(states),) or episodes.shape != clips.shape:
        raise ValueError("clip_ids and episode_ids must have one identifier per transition")
    if set(clips) & (set(fit.train_clip_ids) | set(fit.validation_clip_ids)):
        raise ValueError("evaluation clips overlap training or validation clips")
    predicted, probabilities, spread = predict_next_state(
        fit.model, fit.normalizer, states, actions, hard_contacts=False
    )
    groups = _group_metrics(
        predicted, next_states, states, fit.normalizer, probabilities,
        next_states[:, CONTACT_STATE_INDICES],
    )
    rollout_groups, rollout_contacts, disagreement = _evaluate_rollouts(
        fit.model, states, actions, next_states, clips, episodes,
        np.ones(len(states), dtype=bool), fit.normalizer, rollout_horizon,
    )
    for name in STATE_GROUPS:
        groups[name].update(rollout_groups[name])
    groups.update(rollout_contacts)
    return {
        "transition_count": len(states), "clip_count": len(np.unique(clips)),
        "rollout_horizon_steps": rollout_horizon, "groups": groups,
        "transition_dt_seconds": transition_dt_seconds,
        "rollout_seconds": None if transition_dt_seconds is None else rollout_horizon * transition_dt_seconds,
        "ensemble_disagreement_normalized": disagreement,
        "ensemble_disagreement_one_step_normalized": float(np.mean(spread)),
    }


def save_checkpoint(fit: StateDynamicsFit, destination: str) -> None:
    """Save ensemble weights and training-only normalization outside the repository."""
    if torch is None:
        raise ImportError("saving world-model checkpoints requires PyTorch")
    torch.save({
        "state_dim": STATE_DIM, "action_dim": ACTION_DIM, "hidden_dim": 128,
        "ensemble_size": len(fit.model.members), "model_state_dict": fit.model.state_dict(),
        "normalizer": {name: getattr(fit.normalizer, name) for name in (
            "state_mean", "state_scale", "action_mean", "action_scale", "delta_mean", "delta_scale"
        )},
        "train_clip_ids": fit.train_clip_ids,
        "validation_clip_ids": fit.validation_clip_ids,
    }, destination)


def load_checkpoint(source: str) -> StateDynamicsFit:
    """Load a locally saved V2 checkpoint for evaluation or rollout."""
    if torch is None:
        raise ImportError("loading world-model checkpoints requires PyTorch")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if checkpoint.get("state_dim") != STATE_DIM or checkpoint.get("action_dim") != ACTION_DIM:
        raise ValueError("checkpoint dimensions do not match the V2 state and action schema")
    model = StateDynamicsEnsemble(
        count=int(checkpoint["ensemble_size"]), hidden_dim=int(checkpoint["hidden_dim"])
    )
    incompatibility = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if incompatibility.unexpected_keys or set(incompatibility.missing_keys) - {"residual_gain"}:
        raise ValueError("checkpoint weights do not match the V2 dynamics architecture")
    model.eval()
    normalizer = DynamicsNormalizer(**checkpoint["normalizer"])
    return StateDynamicsFit(
        model=model, normalizer=normalizer,
        train_clip_ids=tuple(checkpoint["train_clip_ids"]),
        validation_clip_ids=tuple(checkpoint["validation_clip_ids"]), metrics={},
    )
