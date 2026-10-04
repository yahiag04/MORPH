"""Action-conditioned ensemble dynamics for the structured 37D Panda state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from world_model.v3.contracts import (
    ACTION_DIMS,
    CONTACT_INDICES,
    DynamicsConfig,
    QUATERNION_SLICE,
    RolloutBatch,
    STATE_DIM,
)

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # V3 data and audit tools remain usable without PyTorch.
    torch = None
    nn = None
    F = None


MODEL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DynamicsNormalizerV3:
    state_mean: np.ndarray
    state_scale: np.ndarray
    action_mean: np.ndarray
    action_scale: np.ndarray
    delta_mean: np.ndarray
    delta_scale: np.ndarray

    def as_dict(self) -> dict[str, list[float]]:
        return {name: getattr(self, name).astype(float).tolist() for name in (
            "state_mean", "state_scale", "action_mean", "action_scale",
            "delta_mean", "delta_scale",
        )}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DynamicsNormalizerV3":
        expected = ("state_mean", "state_scale", "action_mean", "action_scale",
                    "delta_mean", "delta_scale")
        if not isinstance(value, dict) or set(value) != set(expected):
            raise ValueError("checkpoint normalizer fields do not match schema")
        arrays = {name: np.asarray(value[name], dtype=np.float32) for name in expected}
        if (arrays["state_mean"].ndim != 1 or arrays["state_mean"].shape[0] not in (37, 43)
                or arrays["state_scale"].shape != arrays["state_mean"].shape
                or arrays["action_mean"].shape not in ((4,), (8,))
                or arrays["action_scale"].shape != arrays["action_mean"].shape
                or arrays["delta_mean"].shape != (32,) or arrays["delta_scale"].shape != (32,)):
            raise ValueError("checkpoint normalizer has invalid dimensions")
        if not all(np.isfinite(array).all() for array in arrays.values()) or any(
            np.any(arrays[name] <= 0) for name in ("state_scale", "action_scale", "delta_scale")
        ):
            raise ValueError("checkpoint normalizer must contain finite positive scales")
        return cls(**arrays)


if nn is not None:
    class DynamicsMemberV3(nn.Module):
        def __init__(self, input_dim: int, action_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(input_dim + action_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            )
            self.delta_head = nn.Linear(hidden_dim, 32)
            self.contact_head = nn.Linear(hidden_dim, 2)

        def forward(self, state_features, normalized_action):
            hidden = self.trunk(torch.cat((state_features, normalized_action), dim=-1))
            return self.delta_head(hidden), self.contact_head(hidden)


    class DynamicsEnsembleV3(nn.Module):
        def __init__(self, config: DynamicsConfig):
            super().__init__()
            action_dim = ACTION_DIMS[config.action_mode]
            input_dim = STATE_DIM + (6 if config.relative_features else 0)
            self.members = nn.ModuleList(
                DynamicsMemberV3(input_dim, action_dim, config.hidden_dim)
                for _ in range(config.ensemble_size)
            )
else:
    class DynamicsMemberV3:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("DynamicsV3 requires the optional PyTorch runtime")


    class DynamicsEnsembleV3:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("DynamicsV3 requires the optional PyTorch runtime")


class DynamicsV3:
    """Bootstrap ensemble whose members evolve separate state trajectories."""

    def __init__(
        self, config: DynamicsConfig, *, seed: int = 17,
        normalizer: DynamicsNormalizerV3 | None = None,
        training_group_ids: tuple[str, ...] = (),
        validation_group_ids: tuple[str, ...] = (),
    ) -> None:
        if torch is None:
            raise ImportError("DynamicsV3 requires the optional PyTorch runtime")
        if config.action_mode not in ACTION_DIMS:
            raise ValueError(f"unsupported action mode {config.action_mode!r}")
        if config.ensemble_size < 1 or config.hidden_dim < 1:
            raise ValueError("ensemble_size and hidden_dim must be positive")
        if not np.isfinite(config.observation_dt) or config.observation_dt <= 0:
            raise ValueError("observation_dt must be finite and positive")
        self.config = config
        self.normalizer = normalizer
        self.training_group_ids = tuple(sorted(set(map(str, training_group_ids))))
        self.validation_group_ids = tuple(sorted(set(map(str, validation_group_ids))))
        torch.manual_seed(int(seed))
        self.network = DynamicsEnsembleV3(config)
        self.residual_gain = np.ones(32, dtype=np.float32)
        self.training_metadata: dict[str, Any] = {}

    @property
    def device(self):
        return next(self.network.parameters()).device

    def input_features(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != STATE_DIM or not np.isfinite(states).all():
            raise ValueError("states must be a finite [N,37] array")
        if not self.config.relative_features:
            return states.copy()
        package_from_ee = states[:, 17:20] - states[:, 0:3]
        tray_from_package = states[:, 34:37] - states[:, 17:20]
        return np.concatenate((states, package_from_ee, tray_from_package), axis=1)

    def _normalizer_tensors(self, device):
        if self.normalizer is None:
            raise ValueError("model has no fitted training normalizer")
        return {
            name: torch.as_tensor(getattr(self.normalizer, name), dtype=torch.float32, device=device)
            for name in ("state_mean", "state_scale", "action_mean", "action_scale",
                         "delta_mean", "delta_scale")
        }

    def _raw_member(self, member_index: int, states, actions):
        if not 0 <= member_index < len(self.network.members):
            raise IndexError("member_index is outside the ensemble")
        if states.ndim != 2 or states.shape[1] != STATE_DIM:
            raise ValueError("member states must have shape [N,37]")
        action_dim = ACTION_DIMS[self.config.action_mode]
        if actions.ndim != 2 or actions.shape != (len(states), action_dim):
            raise ValueError(f"member actions must have shape [N,{action_dim}]")
        params = self._normalizer_tensors(states.device)
        features = self.input_features(states.detach().cpu().numpy())
        feature_tensor = torch.as_tensor(features, dtype=states.dtype, device=states.device)
        normalized_features = (feature_tensor - params["state_mean"]) / params["state_scale"]
        normalized_actions = (actions - params["action_mean"]) / params["action_scale"]
        return self.network.members[member_index](normalized_features, normalized_actions)

    def predict_member(self, member_index: int, states, actions):
        """Predict next continuous states and return the two contact logits."""
        normalized_delta, contact_logits = self._raw_member(member_index, states, actions)
        params = self._normalizer_tensors(states.device)
        delta = params["delta_mean"] + normalized_delta * params["delta_scale"]
        dynamic = states[:, :32] + torch.as_tensor(
            self.residual_gain, dtype=states.dtype, device=states.device
        ) * delta
        quaternion = dynamic[:, QUATERNION_SLICE]
        norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
        fallback = states[:, QUATERNION_SLICE]
        quaternion = torch.where(norm > 1e-8, quaternion / norm.clamp_min(1e-8), fallback)
        dynamic = torch.cat((dynamic[:, :20], quaternion, dynamic[:, 24:]), dim=-1)
        next_state = torch.cat((dynamic, torch.sigmoid(contact_logits), states[:, 34:37]), dim=-1)
        return next_state, contact_logits

    def rollout(self, initial_states: np.ndarray, actions: np.ndarray) -> RolloutBatch:
        initial = np.asarray(initial_states, dtype=np.float32)
        action = np.asarray(actions, dtype=np.float32)
        action_dim = ACTION_DIMS[self.config.action_mode]
        if initial.ndim != 2 or initial.shape[1] != STATE_DIM or initial.shape[0] < 1:
            raise ValueError("initial_states must have shape [K,37] with K > 0")
        if action.ndim != 3 or action.shape[0] != len(initial) or action.shape[2] != action_dim:
            raise ValueError(f"actions must have shape [K,H,{action_dim}]")
        if action.shape[1] < 1 or not np.isfinite(initial).all() or not np.isfinite(action).all():
            raise ValueError("rollout arrays must be finite and horizon must be positive")
        quaternion_norm = np.linalg.norm(initial[:, QUATERNION_SLICE], axis=1)
        if not np.allclose(quaternion_norm, 1.0, rtol=0.0, atol=1e-3):
            raise ValueError("initial package quaternions must be normalized")
        if not np.isin(initial[:, CONTACT_INDICES], (0.0, 1.0)).all():
            raise ValueError("initial contacts must be binary")
        members, candidates, horizon = len(self.network.members), len(initial), len(action[0])
        output_states = np.zeros((members, candidates, horizon + 1, STATE_DIM), dtype=np.float32)
        output_contacts = np.full((members, candidates, horizon, 2), 0.5, dtype=np.float32)
        valid = np.ones((members, candidates), dtype=bool)
        self.network.eval()
        with torch.no_grad():
            for member_index in range(members):
                current = torch.as_tensor(initial.copy(), dtype=torch.float32, device=self.device)
                output_states[member_index, :, 0] = initial
                for step in range(horizon):
                    active = torch.as_tensor(valid[member_index], dtype=torch.bool, device=self.device)
                    next_state = current.clone()
                    if active.any():
                        action_tensor = torch.as_tensor(action[:, step], dtype=torch.float32,
                                                        device=self.device)
                        predicted, _ = self.predict_member(
                            member_index, current[active], action_tensor[active]
                        )
                        finite = torch.isfinite(predicted).all(dim=1)
                        active_indices = torch.nonzero(active, as_tuple=False).flatten()
                        valid_indices = active_indices[finite]
                        invalid_indices = active_indices[~finite]
                        if len(valid_indices):
                            next_state[valid_indices] = predicted[finite]
                            output_contacts[member_index, valid_indices.cpu().numpy(), step] = (
                                predicted[finite, 32:34].cpu().numpy()
                            )
                        if len(invalid_indices):
                            valid[member_index, invalid_indices.cpu().numpy()] = False
                    current = next_state
                    output_states[member_index, :, step + 1] = current.cpu().numpy()
        return RolloutBatch(output_states, output_contacts, valid)

    def save(
        self, destination: Path, manifest: dict[str, Any], *,
        training_state: dict[str, Any] | None = None,
    ) -> None:
        if self.normalizer is None:
            raise ValueError("cannot save a model without a fitted normalizer")
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "config": asdict(self.config),
            "normalizer": self.normalizer.as_dict(),
            "training_group_ids": list(self.training_group_ids),
            "validation_group_ids": list(self.validation_group_ids),
            "training_metadata": self.training_metadata,
            "manifest": manifest,
            "residual_gain": self.residual_gain.tolist(),
            "state_dict": self.network.state_dict(),
            "training_state": training_state,
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".",
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
            torch.save(payload, temporary)
            os.replace(temporary, target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, source: Path, expected: dict[str, Any] | None = None) -> "DynamicsV3":
        if torch is None:
            raise ImportError("DynamicsV3 checkpoint loading requires PyTorch")
        try:
            checkpoint = torch.load(Path(source), map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError) as error:
            raise ValueError(f"could not load V3 dynamics checkpoint: {error}") from error
        if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ValueError("checkpoint schema version is unsupported")
        config_value = checkpoint.get("config")
        manifest = checkpoint.get("manifest")
        if not isinstance(config_value, dict) or not isinstance(manifest, dict):
            raise ValueError("checkpoint config or manifest is malformed")
        combined = {**config_value, **manifest}
        for key, value in (expected or {}).items():
            if combined.get(key) != value:
                raise ValueError(f"checkpoint {key} does not match expected value")
        try:
            config = DynamicsConfig(**config_value)
            normalizer = DynamicsNormalizerV3.from_dict(checkpoint["normalizer"])
            if len(normalizer.state_mean) != STATE_DIM + (6 if config.relative_features else 0):
                raise ValueError("checkpoint normalizer feature count does not match relative_features")
            if manifest.get("action_mode") != config.action_mode:
                raise ValueError("checkpoint manifest action mode does not match model config")
            model = cls(config, normalizer=normalizer,
                        training_group_ids=tuple(checkpoint["training_group_ids"]),
                        validation_group_ids=tuple(checkpoint.get("validation_group_ids", ())))
            model.network.load_state_dict(checkpoint["state_dict"], strict=True)
            gain = np.asarray(checkpoint["residual_gain"], dtype=np.float32)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"checkpoint contents are incompatible: {error}") from error
        if gain.shape != (32,) or not np.isfinite(gain).all():
            raise ValueError("checkpoint residual gains are malformed")
        if len(model.normalizer.action_mean) != ACTION_DIMS[config.action_mode]:
            raise ValueError("checkpoint normalizer action dimension is incompatible")
        model.residual_gain = gain
        model.training_metadata = checkpoint.get("training_metadata", {})
        model.network.eval()
        return model
