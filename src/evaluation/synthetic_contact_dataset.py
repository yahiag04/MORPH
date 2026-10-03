"""Seeded, local-only contact data collection with grouped scenario partitions."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from evaluation.contact_demo_evaluator import evaluate_contact_trajectory, write_transition_archive
from simulation.contact_manipulation_env import ContactTaskConfig
from simulation.panda_env import DEFAULT_MODEL_PATH

VARIANTS = ('nominal', 'fast', 'grasp_offset', 'release_shift')


def snapshot_robot_model(scene_path: str | Path, destination: str | Path) -> dict:
    """Save the compiled MuJoCo scene and fingerprint it, including XML includes."""
    scene_path = Path(scene_path).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    mujoco.mj_saveModel(model, str(destination))
    return {
        'scene_xml_sha256': hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        'compiled_model_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def make_scenarios(*, families: int = 60, seed: int = 2026) -> list[dict]:
    """Assign source families to partitions before creating any episode variants."""
    if not isinstance(families, int) or families < 4:
        raise ValueError('families must be an integer of at least four')
    if not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    order = np.random.default_rng(np.random.SeedSequence([seed, 911])).permutation(families)
    train_count = min(families - 2, int(families * .70))
    validation_count = max(1, (families - train_count) // 2)
    partitions = {int(index): ('train' if rank < train_count else
                  'validation' if rank < train_count + validation_count else 'test')
                  for rank, index in enumerate(order)}
    scenarios = []
    for family in range(families):
        rng = np.random.default_rng(np.random.SeedSequence([seed, family]))
        pickup = np.array([rng.uniform(.52, .60), rng.uniform(.07, .16)])
        dropoff = np.array([rng.uniform(.50, .61), rng.uniform(-.20, -.11)])
        yaw = float(rng.uniform(-np.pi, np.pi))
        base_speed = float(rng.uniform(.065, .10))
        for variant in VARIANTS:
            pickup_command, dropoff_command = pickup.copy(), dropoff.copy()
            speed = float(rng.uniform(.14, .20)) if variant == 'fast' else base_speed
            release_fraction = float(rng.uniform(.6, 1.0))
            release_height = .52
            if variant == 'grasp_offset':
                angle = rng.uniform(-np.pi, np.pi)
                pickup_command += rng.uniform(.035, .075) * np.array([np.cos(angle), np.sin(angle)])
            if variant == 'release_shift':
                dropoff_command[0] += rng.choice([-1, 1]) * rng.uniform(.13, .18)
                dropoff_command[0] = np.clip(dropoff_command[0], .35, .74)
                release_fraction = float(rng.uniform(0, .4))
                release_height = float(rng.uniform(.55, .62))
            scenarios.append({
                'episode_id': f'synthetic-{seed}-{family:03d}-{variant}',
                'group_id': f'scenario-{seed}-{family:03d}', 'split': partitions[family],
                'source_kind': 'simulation_randomized', 'variant': variant,
                'seed': seed, 'family_index': family,
                'pickup_xy': pickup.tolist(), 'dropoff_xy': dropoff.tolist(),
                'controller_pickup_xy': pickup_command.tolist(),
                'controller_dropoff_xy': dropoff_command.tolist(),
                'package_yaw_radians': yaw, 'cartesian_speed_mps': speed,
                'release_open_fraction': release_fraction, 'release_height_m': release_height,
                'path_bend_m': float(rng.uniform(-.025, .025)),
                'control_physics_steps': 50, 'observation_physics_steps': 10,
            })
    return scenarios


def run_scenario(scenario: dict, *, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """Replay a synthetic path; scenario names do not determine outcome labels."""
    pickup = np.asarray(scenario['controller_pickup_xy'])
    dropoff = np.asarray(scenario['controller_dropoff_xy'])
    approach = np.linspace(np.array([.540, -.050]), pickup, 6)
    carry = np.linspace(pickup, dropoff, 30)[1:]
    carry[:, 0] += scenario['path_bend_m'] * np.sin(np.linspace(0, np.pi, len(carry)))
    xy = np.vstack((approach, carry))
    trajectory = np.column_stack((np.arange(len(xy)) * .1, xy, np.full(len(xy), .62)))
    run = evaluate_contact_trajectory(
        trajectory, scenario['pickup_xy'], scenario['dropoff_xy'], human_label='not_applicable',
        model_path=model_path, simulation_steps_per_sample=scenario['control_physics_steps'],
        observation_steps=scenario['observation_physics_steps'],
        config=ContactTaskConfig(package_yaw_radians=scenario['package_yaw_radians']),
        heights={'approach': .62, 'grasp': .52, 'carry': .70, 'release': scenario['release_height_m']},
        cartesian_speed_mps=scenario['cartesian_speed_mps'],
        release_open_fraction=scenario['release_open_fraction'],
        controller_pickup_xy=pickup, controller_dropoff_xy=dropoff,
    )
    run.update({key: scenario[key] for key in ('episode_id','group_id','split','source_kind','variant')})
    run['clip_id'] = scenario['episode_id']
    return run


def audit_transition_archive(path: Path) -> dict:
    """Check archived temporal/state continuity and report physical event coverage."""
    with np.load(path, allow_pickle=False) as data:
        states, following = data['states'], data['next_states']
        count = len(states)
        if not count or states.shape != (count, 37) or following.shape != states.shape:
            raise ValueError('archive must contain nonempty 37-value transitions')
        for key in ('states','next_states','actions','actuator_controls',
                    'simulation_start_times','simulation_end_times','transition_dt_seconds'):
            if not np.isfinite(data[key]).all():
                raise ValueError(f'nonfinite archive values: {key}')
        if data['actions'].shape != (count,4) or data['actuator_controls'].shape != (count,8):
            raise ValueError('invalid high-level or actuator command shape')
        intervals = data['transition_dt_seconds']
        if np.any(intervals <= 0) or not np.allclose(intervals, intervals[0], rtol=0, atol=1e-9):
            raise ValueError('observation intervals must be positive and uniform')
        if not np.allclose(data['simulation_end_times'] - data['simulation_start_times'], intervals):
            raise ValueError('recorded time interval disagrees with simulation clock')
        if not np.allclose(data['simulation_end_times'][:-1], data['simulation_start_times'][1:]):
            raise ValueError('simulation times contain gaps')
        if not np.array_equal(following[:-1], states[1:]):
            raise ValueError('state transition chain contains gaps')
        for values in (states, following):
            if not np.allclose(np.linalg.norm(values[:,20:24],axis=1),1.,atol=1e-5):
                raise ValueError('package quaternion is not normalized')
            if not np.isin(values[:,32:34],(0,1)).all():
                raise ValueError('contact state is not binary')
        for key in ('episode_ids','group_ids','source_kinds','splits'):
            if len(data[key]) != count or len(np.unique(data[key])) != 1:
                raise ValueError(f'episode provenance is inconsistent: {key}')
        return {
            'transitions': count, 'transition_dt_seconds': float(intervals[0]),
            'simulated_seconds': float(intervals.sum()),
            'phase_counts': dict(Counter(data['phases'].tolist())),
            'gripper_contact_changes': int(np.sum(states[:,32] != following[:,32])),
            'support_contact_changes': int(np.sum(states[:,33] != following[:,33])),
            'package_moving_transitions': int(np.sum(np.linalg.norm(following[:,24:27],axis=1) > .01)),
        }


def collect_dataset(output_dir: Path, *, families: int = 60, seed: int = 2026,
                    model_path: str | Path = DEFAULT_MODEL_PATH,
                    progress: Callable[[str], None] | None = None) -> dict:
    """Collect all planned scenarios, retain failures, and write a reproducible manifest."""
    output_dir = Path(output_dir).expanduser().resolve()
    if any((parent / '.git').exists() for parent in (output_dir, *output_dir.parents)):
        raise ValueError('dataset output must be outside a repository')
    scenarios = make_scenarios(families=families, seed=seed)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('dataset output directory must be empty; existing data will not be overwritten')
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_dir / 'episodes'
    episodes_dir.mkdir()
    source_files = [Path(__file__), Path(__file__).with_name('contact_demo_evaluator.py'),
                    Path(__file__).parents[1]/'simulation'/'demonstration_manipulation.py',
                    Path(__file__).parents[1]/'simulation'/'contact_manipulation_env.py']
    source_content = {p.name: p.read_bytes() for p in source_files}
    source_dir = output_dir / 'collection_source'
    source_dir.mkdir()
    for name, content in source_content.items():
        (source_dir / name).write_bytes(content)
    model_snapshot = snapshot_robot_model(model_path, source_dir / 'robot_model.mjb')
    manifest = {
        'schema_version': 3, 'source_kind': 'simulation_randomized',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': seed,
        'families': families, 'planned_episodes': len(scenarios),
        'mujoco_version': mujoco.__version__, 'numpy_version': np.__version__,
        'scene_path': str(model_path), **model_snapshot,
        'model_snapshot_path': 'collection_source/robot_model.mjb',
        'collection_source_sha256': {name: hashlib.sha256(raw).hexdigest() for name,raw in source_content.items()},
        'collection_source_directory': 'collection_source',
        'fixed_task_config': {k:v for k,v in asdict(ContactTaskConfig()).items() if k != 'package_yaw_radians'},
        'partition_policy': 'Scenario families assigned before simulation; all four interventions stay together.',
        'episodes': [],
    }
    phase_counts, split_counts, failures, variants = Counter(), Counter(), Counter(), {}
    total_transitions = total_successes = collection_errors = contact_changes = moving = 0
    simulated_seconds = 0.
    observation_intervals, control_intervals = set(), set()
    for index, scenario in enumerate(scenarios):
        entry = {'scenario': scenario}
        try:
            run = run_scenario(scenario, model_path=model_path)
            relative_path = Path('episodes') / f"{scenario['episode_id']}.npz"
            write_transition_archive([run], output_dir / relative_path)
            audit = audit_transition_archive(output_dir / relative_path)
            entry.update(archive=str(relative_path), audit=audit, robot_success=run['robot_success'],
                         failure_reason=run['failure_reason'], max_lift_m=run['max_lift_m'])
            total_transitions += audit['transitions']
            simulated_seconds += audit['simulated_seconds']
            observation_intervals.add(round(audit['transition_dt_seconds'], 12))
            control_intervals.add(round(run['control_dt_seconds'], 12))
            total_successes += int(run['robot_success'])
            contact_changes += audit['gripper_contact_changes'] + audit['support_contact_changes']
            moving += audit['package_moving_transitions']
            phase_counts.update(audit['phase_counts'])
            split_counts[scenario['split']] += 1
            if run['failure_reason']:
                failures[run['failure_reason']] += 1
            stats = variants.setdefault(scenario['variant'], {'episodes':0, 'robot_successes':0})
            stats['episodes'] += 1
            stats['robot_successes'] += int(run['robot_success'])
        except (ValueError, RuntimeError, OSError) as error:
            entry['collection_error'] = str(error)
            collection_errors += 1
        manifest['episodes'].append(entry)
        (output_dir / 'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        if progress is not None and ((index+1)%10 == 0 or index+1 == len(scenarios)):
            progress(f'{index+1}/{len(scenarios)} episodes; {total_transitions} transitions; '
                     f'{total_successes} completed transfers; {collection_errors} collection errors')
    summary = {
        'source_kind': 'simulation_randomized', 'human_demonstrations_added': 0,
        'scenario_families': families, 'planned_episodes': len(scenarios),
        'collected_episodes': sum(split_counts.values()), 'collection_errors': collection_errors,
        'episodes_by_split': dict(split_counts), 'transitions': total_transitions,
        'observation_dt_seconds': next(iter(observation_intervals)) if len(observation_intervals) == 1 else None,
        'control_dt_seconds': next(iter(control_intervals)) if len(control_intervals) == 1 else None,
        'simulated_seconds': simulated_seconds, 'robot_successes': total_successes,
        'failure_reasons': dict(failures), 'episodes_by_variant': variants,
        'phase_counts': dict(phase_counts), 'contact_changes': contact_changes,
        'package_moving_transitions': moving,
        'interpretation': 'Coverage of a synthetic collection policy, not learned-controller performance. '
                          'Object mass, friction and geometry are fixed; positions, yaw and commands vary. '
                          'The test partition has not been used for model selection.',
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary
