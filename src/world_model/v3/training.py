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
    "device": "cpu",
    "observation_dt": 0.02,
    "action_mode": "actuator8",
}
TRAINING_SCHEMA_VERSION = 1


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


def _window_loss(member, states, actions, next_states, config, params, *, train: bool):
    current = states[:, 0]
    one_step = []
    rollout = []
    contact_columns = torch.as_tensor(CONTACT_INDICES, dtype=torch.long, device=states.device)
    continuous_scale = params["state_scale"][:32]
    for offset in range(states.shape[1]):
        truth = next_states[:, offset]
        prediction, normalized_delta, contact_logits, contact_probs = _step(
            member, current, actions[:, offset], config, params,
        )
        target_delta = (
            truth[:, :32] - current[:, :32] - params["delta_mean"]
        ) / params["delta_scale"]
        one_step.append(
            F.mse_loss(normalized_delta, target_delta)
            + F.binary_cross_entropy_with_logits(contact_logits, truth[:, contact_columns])
        )
        current = prediction
        if offset + 1 in (1, 5, 25):
            state_loss = F.mse_loss(
                (current[:, :32] - truth[:, :32]) / continuous_scale,
                torch.zeros_like(current[:, :32]),
            )
            contact_loss = F.binary_cross_entropy(
                contact_probs.clamp(1e-6, 1 - 1e-6), truth[:, contact_columns],
            )
            rollout.append(state_loss + contact_loss)
    return 0.5 * torch.stack(one_step).mean() + 0.5 * torch.stack(rollout).mean()


def _make_windows(batch: EpisodeBatch, horizon: int, stride: int) -> list[tuple[str, str, np.ndarray]]:
    windows = []
    for episode, family, rows in _episode_rows(batch):
        for start in range(0, len(rows) - horizon + 1, stride):
            windows.append((episode, family, rows[start:start + horizon]))
    return windows


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
    train_windows = _make_windows(train, resolved["rollout_horizon"], resolved["window_stride"])
    validation_windows = _make_windows(
        validation, resolved["rollout_horizon"], resolved["window_stride"],
    )
    if not train_windows or not validation_windows:
        raise ValueError("train and validation episodes need a full rollout window")

    destination = Path(output_dir).expanduser().resolve()
    resolved["seed"] = int(resolved["seed"])
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
    train_by_family: dict[str, list[np.ndarray]] = {group: [] for group in train_groups}
    for _, family, rows in train_windows:
        train_by_family[family].append(rows)
    family_ids = np.asarray(sorted(train_by_family))
    logs = destination / "training.jsonl"
    best_losses = [] if prior_state is None else list(prior_state["best_losses"])
    best_families = [] if prior_state is None else list(prior_state["best_families"])
    epochs_run = [] if prior_state is None else list(prior_state["epochs_run"])
    start_member = 0 if prior_state is None else int(prior_state["member_index"])
    if prior_state is not None:
        model.network.load_state_dict(prior_state["network_state"], strict=True)

    def save_progress(member_index, next_epoch, optimizer_state, member_rng_state,
                      best_state, best_loss, best_breakdown, stale):
        state = {
            "status": "running", "member_index": member_index,
            "next_epoch": next_epoch, "optimizer_state": optimizer_state,
            "member_rng_state": member_rng_state, "best_state": best_state,
            "best_loss": best_loss, "best_breakdown": best_breakdown,
            "stale": stale, "best_losses": list(best_losses),
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
            bootstrap_windows = [window for family in bootstrap for window in train_by_family[str(family)]]
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
            if prior_state is not None and member_index == start_member:
                epoch_start = int(prior_state["next_epoch"])
                if prior_state["optimizer_state"] is not None:
                    member_rng.bit_generator.state = prior_state["member_rng_state"]
                    optimizer.load_state_dict(prior_state["optimizer_state"])
                    best_state = prior_state["best_state"]
                    best_loss = float(prior_state["best_loss"])
                    best_breakdown = prior_state["best_breakdown"]
                    stale = int(prior_state["stale"])
                    completed = epoch_start
                    torch.set_rng_state(prior_state["torch_rng_state"])
                prior_state = None
            if epoch_start == 0:
                save_progress(member_index, 0, optimizer.state_dict(),
                              member_rng.bit_generator.state, best_state, best_loss,
                              best_breakdown, stale)
            for epoch in range(epoch_start, resolved["epochs"]):
                started = time.perf_counter()
                member.train()
                permutation = member_rng.permutation(len(bootstrap_windows))
                train_losses = []
                for offset in range(0, len(permutation), resolved["batch_size"]):
                    selected = [bootstrap_windows[index]
                                for index in permutation[offset:offset + resolved["batch_size"]]]
                    indices = np.stack(selected)
                    states = torch.as_tensor(train.states[indices], dtype=torch.float32, device=device)
                    actions = torch.as_tensor(train.actions[indices], dtype=torch.float32, device=device)
                    next_states = torch.as_tensor(train.next_states[indices], dtype=torch.float32,
                                                  device=device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = _window_loss(member, states, actions, next_states,
                                        resolved, params, train=True)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("training loss became non-finite")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(member.parameters(), resolved["gradient_clip"])
                    optimizer.step()
                    train_losses.append(float(loss.detach().cpu()))
                member.eval()
                validation_loss, family_losses = _validation_score(
                    member, validation, validation_windows, resolved, params,
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
                        "train_loss": float(np.mean(train_losses)),
                        "validation_family_loss": validation_loss,
                        "validation_family_count": len(family_losses),
                        "seconds": time.perf_counter() - started,
                    }, allow_nan=False) + "\n")
                save_progress(member_index, completed, optimizer.state_dict(),
                              member_rng.bit_generator.state, best_state, best_loss,
                              best_breakdown, stale)
                if stale >= resolved["patience"]:
                    break
            if best_state is None:
                raise RuntimeError("training did not produce a finite validation checkpoint")
            member.load_state_dict(best_state)
            best_losses.append(best_loss)
            best_families.append(best_breakdown)
            epochs_run.append(completed)
            if member_index + 1 < len(model.network.members):
                save_progress(member_index + 1, 0, None, None, None, float("inf"), {}, 0)

        model.network.eval()
        model.save(destination / "best.pt", run_manifest)
        metrics = {
            "schema_version": 1,
            "model": asdict(model_config),
            "epochs_per_member": epochs_run,
            "best_validation_family_loss_per_member": best_losses,
            "best_validation_loss_by_family_per_member": best_families,
            "training_window_count": len(train_windows),
            "validation_window_count": len(validation_windows),
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
