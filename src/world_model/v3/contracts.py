"""Dependency-light data contracts shared by V3 tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


STATE_DIM = 37
CONTACT_INDICES = (32, 33)
QUATERNION_SLICE = slice(20, 24)
ACTION_DIMS = {"cartesian4": 4, "actuator8": 8}


@dataclass(frozen=True)
class DynamicsConfig:
    """Versioned predictor input and training dimensions."""

    action_mode: str
    relative_features: bool = False
    ensemble_size: int = 3
    hidden_dim: int = 128
    observation_dt: float = 0.02


@dataclass
class EpisodeBatch:
    """Aligned transitions with their complete episode and family provenance."""

    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    episode_ids: np.ndarray
    group_ids: np.ndarray
    phases: np.ndarray
    start_times: np.ndarray
    end_times: np.ndarray
    metadata: dict[str, Any]


@dataclass
class RolloutBatch:
    """Member- and candidate-specific imagined futures."""

    states: np.ndarray
    contact_probs: np.ndarray
    valid: np.ndarray


@dataclass
class PlannerDecision:
    """The command to apply and diagnostics for one receding-horizon step."""

    control: np.ndarray
    fallback: bool
    diagnostics: dict[str, Any]
