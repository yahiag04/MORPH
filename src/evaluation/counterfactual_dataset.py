"""Collect replayable, paired MuJoCo interventions from common initial scenes."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import tempfile
from typing import Callable

import mujoco
import numpy as np

from evaluation.contact_demo_evaluator import contact_task_state, evaluate_contact_trajectory, write_transition_archive
from evaluation.synthetic_contact_dataset import audit_transition_archive
from simulation.contact_manipulation_env import ContactManipulationEnv, ContactTaskConfig
from simulation.panda_env import DEFAULT_MODEL_PATH


VARIANTS = (
    "nominal", "speed_1_5x", "speed_0_7x", "pickup_x_plus_15mm",
    "pickup_x_minus_15mm", "pickup_x_plus_30mm", "release_early", "release_late",
)
SNAPSHOT_SCHEMA_VERSION = 2
OBSERVATION_STEPS = 10
CONTROL_STEPS = 50


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compiled_model_fingerprint(model: mujoco.MjModel) -> str:
    """Hash the compiled model without retaining its large embedded mesh assets."""
    with tempfile.TemporaryDirectory(prefix="morph-counterfactual-model-") as temporary:
        path = Path(temporary) / "scene.mjb"
        mujoco.mj_saveModel(model, str(path))
        return _sha256_file(path)


def _integration_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    values = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, values, spec)
    return values


def _snapshot_payload(model: mujoco.MjModel, data: mujoco.MjData,
                      compiled_model_sha256: str, scenario: dict,
                      base_model_sha256: str) -> dict:
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "compiled_model_sha256": compiled_model_sha256,
        "base_model_sha256": base_model_sha256,
        "pickup_xy": np.asarray(scenario["pickup_xy"], dtype=np.float64),
        "dropoff_xy": np.asarray(scenario["dropoff_xy"], dtype=np.float64),
        "package_yaw_radians": float(scenario["package_yaw_radians"]),
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
        "nbody": int(model.nbody), "ngeom": int(model.ngeom),
        "integration_state": _integration_state(model, data),
        "qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "time": float(data.time),
        "ctrl": data.ctrl.copy(),
        "qacc_warmstart": data.qacc_warmstart.copy(),
        "mocap_pos": data.mocap_pos.copy(), "mocap_quat": data.mocap_quat.copy(),
        "userdata": data.userdata.copy(), "eq_active": data.eq_active.copy(),
    }


def _save_snapshot(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def load_family_snapshot(
    path: str | Path, *, model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict:
    """Rebuild the small scene model and restore a numeric-only snapshot."""
    snapshot_path = Path(path).expanduser().resolve()
    with np.load(snapshot_path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    if int(values.get("schema_version", -1)) != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported counterfactual snapshot schema")
    base_model = Path(model_path).expanduser().resolve()
    if not base_model.is_file() or _sha256_file(base_model) != str(
        values["base_model_sha256"].item()
    ):
        raise ValueError("snapshot base robot model does not match its recorded fingerprint")
    scenario = {
        "pickup_xy": values["pickup_xy"].astype(np.float64).tolist(),
        "dropoff_xy": values["dropoff_xy"].astype(np.float64).tolist(),
        "package_yaw_radians": float(values["package_yaw_radians"]),
    }
    env = ContactManipulationEnv(
        model_path=base_model,
        pickup_xyz=(*scenario["pickup_xy"], 0.412),
        dropoff_xyz=(*scenario["dropoff_xy"], 0.416),
        config=ContactTaskConfig(package_yaw_radians=scenario["package_yaw_radians"]),
    )
    model = env.model
    if _compiled_model_fingerprint(model) != str(values["compiled_model_sha256"].item()):
        raise ValueError("recompiled scene fingerprint does not match the recorded model")
    payload = {
        name: values[name] for name in (
            "integration_state", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat",
            "userdata", "eq_active", "qpos", "qvel", "time",
        )
    }
    payload.update({
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "compiled_model_sha256": str(values["compiled_model_sha256"].item()),
        "base_model_sha256": str(values["base_model_sha256"].item()),
        "pickup_xy": values["pickup_xy"], "dropoff_xy": values["dropoff_xy"],
        "package_yaw_radians": float(values["package_yaw_radians"]),
        "nq": int(values["nq"]), "nv": int(values["nv"]),
        "nu": int(values["nu"]), "nbody": int(values["nbody"]),
        "ngeom": int(values["ngeom"]),
    })
    data = mujoco.MjData(model)
    restore_family_snapshot(model, data, payload)
    payload["model"] = model
    payload["data"] = data
    payload["snapshot_path"] = str(snapshot_path)
    payload["snapshot_sha256"] = _sha256_file(snapshot_path)
    return payload


def restore_family_snapshot(
    model: mujoco.MjModel, data: mujoco.MjData, snapshot: dict,
) -> None:
    """Restore integration state plus controls and solver warmstart."""
    required = ("integration_state", "ctrl", "qacc_warmstart", "mocap_pos",
                "mocap_quat", "userdata", "eq_active")
    if any(key not in snapshot for key in required):
        raise ValueError("snapshot is missing full-state integration fields")
    dimensions = (int(model.nq), int(model.nv), int(model.nu), int(model.nbody), int(model.ngeom))
    recorded = tuple(int(snapshot[name]) for name in ("nq", "nv", "nu", "nbody", "ngeom"))
    if dimensions != recorded:
        raise ValueError("snapshot dimensions do not match compiled MuJoCo model")
    for name in required:
        if not np.isfinite(snapshot[name]).all():
            raise ValueError(f"snapshot field {name} contains non-finite values")
    mujoco.mj_resetData(model, data)
    mujoco.mj_setState(
        model, data, np.asarray(snapshot["integration_state"], dtype=np.float64),
        mujoco.mjtState.mjSTATE_INTEGRATION,
    )
    for name in ("ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat", "userdata", "eq_active"):
        target = getattr(data, name)
        value = np.asarray(snapshot[name])
        if target.shape != value.shape:
            raise ValueError(f"snapshot field {name} has a shape mismatch")
        target[:] = value
    mujoco.mj_forward(model, data)


def capture_family_snapshot(scenario: dict, *, output_dir: Path,
                            model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """Create one physical initial state for all interventions in a family."""
    output_dir = Path(output_dir).expanduser().resolve()
    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    group_id = str(scenario["group_id"])
    snapshot_file = snapshot_dir / f"{group_id}.npz"
    if snapshot_file.exists():
        raise ValueError(f"snapshot already exists for family {group_id}")
    config = ContactTaskConfig(package_yaw_radians=float(scenario["package_yaw_radians"]))
    env = ContactManipulationEnv(
        model_path=model_path,
        pickup_xyz=(*scenario["pickup_xy"], 0.412),
        dropoff_xyz=(*scenario["dropoff_xy"], 0.416), config=config,
    )
    compiled_hash = _compiled_model_fingerprint(env.model)
    clone = mujoco.MjData(env.model)
    mujoco.mj_copyData(clone, env.model, env.data)
    payload = _snapshot_payload(
        env.model, clone, compiled_hash, scenario, _sha256_file(Path(model_path).expanduser().resolve()),
    )
    _save_snapshot(payload, snapshot_file)
    loaded = load_family_snapshot(snapshot_file)
    return {
        "group_id": group_id, "snapshot_path": str(snapshot_file),
        "snapshot_sha256": loaded["snapshot_sha256"],
        "compiled_model_sha256": compiled_hash,
        "initial_data": clone,
        "initial_state_37": contact_task_state(env).tolist(),
        "time": float(clone.time),
    }


def _make_trajectory(scenario: dict) -> np.ndarray:
    pickup = np.asarray(scenario["controller_pickup_xy"], dtype=np.float64)
    dropoff = np.asarray(scenario["controller_dropoff_xy"], dtype=np.float64)
    approach = np.linspace(np.array([0.540, -0.050]), pickup, 6)
    carry = np.linspace(pickup, dropoff, 30)[1:]
    carry[:, 0] += float(scenario["path_bend_m"]) * np.sin(np.linspace(0.0, np.pi, len(carry)))
    xy = np.vstack((approach, carry))
    return np.column_stack((np.arange(len(xy)) * 0.1, xy, np.full(len(xy), 0.62)))


def make_counterfactual_families(
    *, families: int, seed: int, split: str,
) -> list[dict]:
    """Create 8 single-intervention variants from each fixed family scene."""
    if not isinstance(families, int) or families < 4:
        raise ValueError("families must be an integer of at least four")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if split not in ("development", "test"):
        raise ValueError("split must be development or test")
    order = np.random.default_rng(np.random.SeedSequence([seed, 410])).permutation(families)
    train_count = int(0.8 * families)
    family_splits = {
        int(index): ("test" if split == "test" else
                     "train" if rank < train_count else "validation")
        for rank, index in enumerate(order)
    }
    result = []
    for family in range(families):
        rng = np.random.default_rng(np.random.SeedSequence([seed, family, 81]))
        pickup = np.array([rng.uniform(0.52, 0.60), rng.uniform(0.07, 0.16)])
        dropoff = np.array([rng.uniform(0.50, 0.61), rng.uniform(-0.20, -0.11)])
        nominal_speed = float(rng.uniform(0.065, 0.10))
        yaw = float(rng.uniform(-np.pi, np.pi))
        path_bend = float(rng.uniform(-0.025, 0.025))
        group_id = f"counterfactual-{seed}-{family:03d}"
        interventions = {
            "nominal": (0.0, nominal_speed, 0.0, "nominal"),
            "speed_1_5x": (0.0, nominal_speed * 1.5, 0.0, "speed"),
            "speed_0_7x": (0.0, nominal_speed * 0.7, 0.0, "speed"),
            "pickup_x_plus_15mm": (0.015, nominal_speed, 0.0, "pickup_x"),
            "pickup_x_minus_15mm": (-0.015, nominal_speed, 0.0, "pickup_x"),
            "pickup_x_plus_30mm": (0.030, nominal_speed, 0.0, "pickup_x"),
            "release_early": (0.0, nominal_speed, 0.2, "release_open_fraction"),
            "release_late": (0.0, nominal_speed, 0.9, "release_open_fraction"),
        }
        for variant in VARIANTS:
            offset, speed, release_fraction, intervention = interventions[variant]
            controller_pickup = pickup + np.asarray([offset, 0.0])
            result.append({
                "episode_id": f"{group_id}-{variant}", "group_id": group_id,
                "split": family_splits[family], "source_kind": "simulation_randomized",
                "dataset_kind": "counterfactual_intervention", "variant": variant,
                "intervention": intervention, "intervention_value": float(
                    offset if intervention == "pickup_x" else
                    speed / nominal_speed if intervention == "speed" else release_fraction
                ),
                "family_index": family, "seed": seed,
                "pickup_xy": pickup.tolist(), "dropoff_xy": dropoff.tolist(),
                "controller_pickup_xy": controller_pickup.tolist(),
                "controller_dropoff_xy": dropoff.tolist(),
                "package_yaw_radians": yaw, "cartesian_speed_mps": speed,
                "release_open_fraction": release_fraction, "release_height_m": 0.52,
                "path_bend_m": path_bend,
                "control_physics_steps": CONTROL_STEPS,
                "observation_physics_steps": OBSERVATION_STEPS,
            })
    return result


def run_counterfactual_scenario(
    scenario: dict, *, initial_snapshot: mujoco.MjData,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    compiled_model: mujoco.MjModel | None = None,
) -> dict:
    """Run one intervention from a cloned family MjData, retaining its outcome."""
    trajectory = _make_trajectory(scenario)
    run = evaluate_contact_trajectory(
        trajectory, scenario["pickup_xy"], scenario["dropoff_xy"],
        human_label="not_applicable", model_path=model_path,
        simulation_steps_per_sample=int(scenario["control_physics_steps"]),
        observation_steps=int(scenario["observation_physics_steps"]),
        config=ContactTaskConfig(package_yaw_radians=float(scenario["package_yaw_radians"])),
        heights={"approach": 0.62, "grasp": 0.52, "carry": 0.70, "release": 0.52},
        cartesian_speed_mps=float(scenario["cartesian_speed_mps"]),
        release_open_fraction=float(scenario["release_open_fraction"]),
        controller_pickup_xy=np.asarray(scenario["controller_pickup_xy"]),
        controller_dropoff_xy=np.asarray(scenario["controller_dropoff_xy"]),
        compiled_model=compiled_model,
        initial_snapshot=initial_snapshot,
    )
    run.update({key: scenario[key] for key in (
        "episode_id", "group_id", "split", "source_kind", "variant",
    )})
    run["clip_id"] = scenario["episode_id"]
    run["intervention"] = scenario["intervention"]
    run["intervention_value"] = scenario["intervention_value"]
    return run


def replay_control_tape(
    run: dict, *, snapshot: dict, scenario: dict, model_path: str | Path = DEFAULT_MODEL_PATH,
    compiled_model: mujoco.MjModel | None = None, atol: float = 1e-5,
) -> dict:
    """Replay archived actuator controls from the identical full simulator state."""
    config = ContactTaskConfig(package_yaw_radians=float(scenario["package_yaw_radians"]))
    env = ContactManipulationEnv(
        model_path=model_path,
        pickup_xyz=(*scenario["pickup_xy"], 0.412),
        dropoff_xyz=(*scenario["dropoff_xy"], 0.416), config=config,
        model=compiled_model,
    )
    restore_family_snapshot(env.model, env.data, snapshot)
    observation_data = mujoco.MjData(env.model)
    max_error = 0.0
    for index, transition in enumerate(run["transitions"]):
        controls = np.asarray(transition["actuator_controls"], dtype=np.float64)
        if controls.shape != (env.model.nu,) or not np.isfinite(controls).all():
            raise ValueError("recorded control tape does not match the MuJoCo actuator layout")
        env.data.ctrl[:] = controls
        duration = float(transition["dt_seconds"])
        steps = int(round(duration / env.model.opt.timestep))
        if steps < 1 or not np.isclose(steps * env.model.opt.timestep, duration, atol=1e-9):
            raise ValueError("recorded transition duration is not an integer number of physics steps")
        for _ in range(steps):
            env.step()
        actual = contact_task_state(env, observation_data=observation_data)
        error = float(np.max(np.abs(actual - transition["next_state"])))
        max_error = max(max_error, error)
        if not np.allclose(actual, transition["next_state"], rtol=0.0, atol=atol):
            raise ValueError(f"control-tape replay diverged at transition {index}: max error {error}")
    return {"verified": True, "max_abs_state_error": max_error,
            "transition_count": len(run["transitions"]), "tolerance": float(atol)}


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def collect_counterfactual_dataset(
    output_dir: Path, scenarios: list[dict], *, model_path: str | Path = DEFAULT_MODEL_PATH,
    progress: Callable[[str], None] | None = None, verify_replay: bool = True,
) -> dict:
    """Collect paired outcomes, snapshots, control tapes, and audit records."""
    output_dir = Path(output_dir).expanduser().resolve()
    if any((parent / ".git").exists() for parent in (output_dir, *output_dir.parents)):
        raise ValueError("counterfactual dataset output must be outside a repository")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("counterfactual dataset output directory must be empty")
    if not scenarios:
        raise ValueError("at least one intervention scenario is required")
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    groups: dict[str, list[dict]] = defaultdict(list)
    for scenario in scenarios:
        if scenario.get("variant") not in VARIANTS:
            raise ValueError("scenario uses an unsupported intervention variant")
        groups[str(scenario["group_id"])].append(scenario)
    for group_id, variants in groups.items():
        if len(variants) != len(VARIANTS) or {item["variant"] for item in variants} != set(VARIANTS):
            raise ValueError(f"family {group_id} must contain each of the eight variants exactly once")
        if len({item["split"] for item in variants}) != 1:
            raise ValueError(f"counterfactual family {group_id} crosses data partitions")
        if len({item["episode_id"] for item in variants}) != len(VARIANTS):
            raise ValueError(f"counterfactual family {group_id} has duplicate episode identifiers")

    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir()
    manifest = {
        "schema_version": 3, "source_kind": "simulation_randomized",
        "dataset_kind": "counterfactual_intervention", "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(scenarios[0]["seed"]), "families": len(groups),
        "planned_episodes": len(scenarios), "variant_names": list(VARIANTS),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "mujoco": mujoco.__version__},
        "rng_policy": "NumPy SeedSequence([seed, family_index, 81]) creates family geometry and path; "
                      "SeedSequence([seed, 410]) determines the family-level split.",
        "scenario_generation": {
            "pickup_x_m": [0.52, 0.60], "pickup_y_m": [0.07, 0.16],
            "tray_x_m": [0.50, 0.61], "tray_y_m": [-0.20, -0.11],
            "nominal_speed_mps": [0.065, 0.10], "yaw_rad": [-float(np.pi), float(np.pi)],
            "path_bend_m": [-0.025, 0.025],
        },
        "fixed_physical_parameters": {
            "physics_timestep_seconds": 0.002, "observation_physics_steps": OBSERVATION_STEPS,
            "control_physics_steps": CONTROL_STEPS, "release_height_m": 0.52,
        },
        "split_counts": dict(Counter(item["split"] for item in scenarios)),
        "partition_policy": "80% train and 20% validation by family before collection; all test families held out.",
        "snapshot_format": "numeric mjSTATE_INTEGRATION plus controls, solver warmstart, mocap, equality and userdata; model rebuilt and fingerprint-checked from recorded scene parameters",
        "replay_tolerance_state_max_abs": 1e-5,
        "replay_verified": bool(verify_replay), "episodes": [], "families_manifest": [],
    }
    source_files = [
        Path(__file__), Path(__file__).with_name("contact_demo_evaluator.py"),
        Path(__file__).with_name("synthetic_contact_dataset.py"),
        Path(__file__).parents[1] / "simulation" / "contact_manipulation_env.py",
        Path(__file__).parents[1] / "simulation" / "demonstration_manipulation.py",
        Path(__file__).parents[1] / "simulation" / "panda_env.py",
        Path(__file__).parents[1] / "simulation" / "ik.py",
        Path(__file__).parents[2] / "scripts" / "collect_counterfactual_dataset.py",
    ]
    manifest["collector_source_sha256"] = {
        str(path.relative_to(Path(__file__).parents[2])): _sha256_file(path)
        for path in source_files
    }
    manifest["robot_model_sha256"] = _sha256_file(model_path)
    split_counts, successes, collection_errors = Counter(), Counter(), Counter()
    transitions_total = 0
    for family_index, (group_id, family_scenarios) in enumerate(sorted(groups.items())):
        try:
            snapshot_meta = capture_family_snapshot(
                family_scenarios[0], output_dir=output_dir, model_path=model_path,
            )
            snapshot = load_family_snapshot(snapshot_meta["snapshot_path"])
            manifest["families_manifest"].append({
                "group_id": group_id, "split": family_scenarios[0]["split"],
                "snapshot_path": str(Path("snapshots") / f"{group_id}.npz"),
                "snapshot_sha256": snapshot_meta["snapshot_sha256"],
                "compiled_model_sha256": snapshot_meta["compiled_model_sha256"],
                "initial_state_37": snapshot_meta["initial_state_37"],
            })
            for scenario in sorted(family_scenarios, key=lambda item: VARIANTS.index(item["variant"])):
                entry = {"scenario": scenario, "snapshot_sha256": snapshot_meta["snapshot_sha256"]}
                try:
                    run = run_counterfactual_scenario(
                        scenario, initial_snapshot=snapshot["data"], model_path=model_path,
                        compiled_model=snapshot["model"],
                    )
                    if verify_replay:
                        entry["replay_audit"] = replay_control_tape(
                            run, snapshot=snapshot, scenario=scenario, model_path=model_path,
                            compiled_model=snapshot["model"],
                        )
                    archive_rel = Path("episodes") / f"{scenario['episode_id']}.npz"
                    write_transition_archive([run], output_dir / archive_rel)
                    audit = audit_transition_archive(output_dir / archive_rel)
                    entry.update({
                        "archive": str(archive_rel), "audit": audit,
                        "robot_success": run["robot_success"],
                        "failure_reason": run["failure_reason"],
                        "max_lift_m": run["max_lift_m"],
                    })
                    split_counts[scenario["split"]] += 1
                    successes[scenario["variant"]] += int(run["robot_success"])
                    transitions_total += audit["transitions"]
                except (ValueError, RuntimeError, OSError) as error:
                    entry["collection_error"] = str(error)
                    collection_errors[type(error).__name__] += 1
                manifest["episodes"].append(entry)
                _atomic_json(output_dir / "manifest.json", manifest)
            if progress is not None:
                progress(f"{family_index + 1}/{len(groups)} families; "
                         f"{transitions_total} transitions; {sum(successes.values())} successes; "
                         f"{sum(collection_errors.values())} collection errors")
        except (ValueError, RuntimeError, OSError) as error:
            collection_errors[type(error).__name__] += 1
            for scenario in family_scenarios:
                manifest["episodes"].append({"scenario": scenario, "collection_error": str(error)})
            _atomic_json(output_dir / "manifest.json", manifest)

    summary = {
        "source_kind": "simulation_randomized", "dataset_kind": "counterfactual_intervention",
        "families": len(groups), "planned_episodes": len(scenarios),
        "collected_episodes": sum(split_counts.values()),
        "collection_errors": sum(collection_errors.values()),
        "collection_errors_by_type": dict(collection_errors),
        "episodes_by_split": dict(split_counts), "transitions": transitions_total,
        "successes_by_variant": dict(successes),
        "replay_verified": bool(verify_replay),
        "interpretation": "Each intervention starts from the same captured MuJoCo MjData within its family. "
                          "Success/failure labels are outcomes, never scenario inputs.",
    }
    _atomic_json(output_dir / "summary.json", summary)
    manifest["summary"] = summary
    _atomic_json(output_dir / "manifest.json", manifest)
    return summary
