# Recording-Driven Contact Manipulation Implementation Plan

**Goal:** Make recorded human demonstrations drive a physical Panda pick-and-place in MuJoCo, then train and evaluate a small action-conditioned world model on the resulting task-state transitions.

**Architecture:** Add a separate compact-package-and-tray environment, per-clip object-center detection with a local task-layout fallback, and a demonstration-conditioned finite-state controller. An evaluator replays every tracked clip and logs MuJoCo state/action transitions; an optional PyTorch module learns next-state dynamics from those transitions and reports clip-held-out prediction quality.

**Tech Stack:** Python 3.14 for the existing project, MuJoCo 3.14.0, NumPy, OpenCV, and an isolated Python/PyTorch environment for world-model training.

**Design:** `docs/designs/recording-driven-contact-manipulation.md`

## Global Constraints

- Video, task-layout coordinates, per-frame paths, local simulation logs, transition datasets, checkpoints, and per-clip predictions stay outside Git.
- Only the human-recorded XY path supplies robot motion; grasp and place heights are declared assumptions because the recordings are overhead.
- The package and receiving tray are simulation proxies; do not present the trial as real hardware validation.
- Physical object movement must come from MuJoCo contact dynamics; do not teleport, weld, or attach the package.
- Keep human attempt labels separate from simulated robot outcomes.
- The world model predicts the next simulated state from current state and commanded action. Report results against a persistence baseline and do not call it a VLA or trained manipulation policy.

## File Map

- Create `src/simulation/contact_manipulation_env.py`: Panda, table, compact free package, and collision-enabled receiving tray.
- Extend `src/perception/task_layout.py`: validated JSON fallback and per-frame package/bowl center detection in rectified unit-square coordinates.
- Create `src/simulation/demonstration_manipulation.py`: deterministic approach/grasp/carry/release controller driven by retargeted human XY.
- Create `src/evaluation/contact_demo_evaluator.py`: fresh seeded simulation per clip, physical success metrics, and transition logs.
- Create `scripts/evaluate_contact_demos.py`: process the local manifest/calibrations/layout, run both retargeting variants, and save private per-clip data plus aggregate-only output.
- Create `src/world_model/state_dynamics.py`: normalized PyTorch one-step action-conditioned state predictor and rollout metrics.
- Create `scripts/train_world_model.py`: train the predictor from local transition NPZ files, with clip-level validation split and checkpoint kept local.
- Create `tests/test_contact_manipulation_env.py`, `tests/test_demonstration_manipulation.py`, `tests/test_contact_demo_evaluator.py`, and `tests/test_state_dynamics.py`.
- Update `README.md` and `docs/data_collection.md`; add aggregate metrics/plots under `results/` only after running the experiment.

---

## Task 1: Build the Contact Task Scene

**Files:** create `src/simulation/contact_manipulation_env.py` and `tests/test_contact_manipulation_env.py`.

- [ ] Add failing tests for the package free joint, package dimensions/initial pose, tray floor/wall collision geometry, named Panda gripper actuator, and rejection of a non-finite or out-of-workspace layout position.
- [ ] Run `PYTHONPATH=src ~/Documents/.venv/bin/python -m unittest tests.test_contact_manipulation_env -v`; confirm the module is missing.
- [ ] Implement `ContactManipulationEnv(model_path=DEFAULT_MODEL_PATH, *, pickup_xyz, dropoff_xyz, config=None)` as a separate `PandaEnv` scene. Use a compact rectangular rigid box (default half-sizes 0.025, 0.020, 0.012 m, mass 0.06 kg) and a square shallow tray with a colliding floor and four short walls.
- [ ] Add methods for package pose/velocity, named gripper open/close, package-to-gripper contact, support contact with table/tray, and stable placement inside the tray.
- [ ] Run focused tests and confirm the existing oracle manipulation tests still pass unchanged.
- [ ] Commit as `feat: add package contact task scene`.

## Task 2: Retarget Demonstrations Into Contact Actions

**Files:** create `src/perception/task_layout.py`, `src/simulation/demonstration_manipulation.py`, and `tests/test_demonstration_manipulation.py`.

- [ ] Add failing layout tests for JSON round-trip, unit-square validation, and conversion of pickup/dropoff points into robot XY using the configured workspace bounds.
- [ ] Add a failing state-machine test with a synthetic path that enters pickup then dropoff regions; assert gripper commands occur in order and package motion comes from contacts.
- [ ] Run the focused tests and confirm failure before implementation.
- [ ] Implement `TaskLayout.load(path)` for JSON fields `pickup_uv` and `dropoff_uv`, each a pair in `[0, 1]` rectified workspace coordinates.
- [ ] Implement `run_demonstration_manipulation(env, timed_xy, layout_xy, *, heights, radii, on_control_step=None)` to approach at z=0.52 m, grasp at table surface +0.035 m, carry at z=0.56 m, and release at table surface +0.035 m. Use pickup/dropoff trigger radii 0.06 m and 0.08 m; retain a closed gripper after grasp and stop the episode as failed if a phase trigger is never reached.
- [ ] Log each control step's timestamp, controller phase, target XYZ, actual XYZ, gripper command, package pose/velocity, contact flags, and IK status.
- [ ] Run focused tests plus the existing full suite; verify no teleportation and repeatable reset behavior.
- [ ] Commit as `feat: control package transfer from demonstrations`.

## Task 3: Evaluate the 16 Real Demonstrations in MuJoCo

**Files:** create `src/evaluation/contact_demo_evaluator.py`, `scripts/evaluate_contact_demos.py`, and `tests/test_contact_demo_evaluator.py`.

- [x] Add failing tests proving deterministic fresh runs, explicit phase-trigger failures, stable physical success criteria, human-label/robot-outcome separation, and aggregate denominator includes failures.
- [x] Implement `evaluate_contact_trajectory(trajectory, pickup_xy, dropoff_xy, model_path=None)` to return per-step state/action transitions and robot outcome metrics.
- [x] Implement CLI inputs `--processing-manifest`, `--calibration-dir`, optional `--task-layout` fallback, and `--output-dir`; output must resolve outside the repo. Run direct and confidence-aware methods for all clips and keep details local.
- [x] Aggregate completed runs, simulation success rate by human label, human/simulation confusion counts, lift/place success, IK failures, and transition counts. Retain failed and incomplete episodes in denominators.
- [ ] Generate an aggregate-only comparison figure and an optional local paired clip. Do not publish video frames, clip names, layout points, or per-clip data.
- [x] Verify on all 16 real clips and both retargeting methods. Direct: 5/12 human-success clips and 0/4 human-failure clips completed; confidence-aware: 4/12 and 0/4. Every clip was retained in denominators. Inspect both successful and failed runs; report the proxy and hand-offset limits.
- [x] Commit as `feat: evaluate contact manipulation from videos`.

## Task 4: Train and Evaluate the State World Model

**Files:** create `src/world_model/state_dynamics.py`, `scripts/train_world_model.py`, `tests/test_state_dynamics.py`, and an optional training requirements file.

- [x] Add failing tests for state/action dimensions, finite normalized inputs, prediction shape, clip-level train/validation split, and improvement reporting relative to persistence on a deterministic synthetic transition set.
- [x] Select an available supported Python/PyTorch runtime and document it separately from the existing Python 3.14 environment; do not add PyTorch to core requirements.
- [x] Implement a small MLP transition model with input `[state_t, action_t]` and target `state_(t+1)-state_t`; include continuous-state normalization calculated only from training clips.
- [x] Train from locally generated MuJoCo transitions using human-derived trajectory commands. Split whole clip identifiers before training so no transitions from one clip cross into both sets. Use the ordered 19-value state `[ee_xyz, arm_qpos, package_xyz, package_linear_velocity, gripper_aperture, package_gripper_contact, package_support_contact]`, 4-value action `[cartesian_target_xyz, gripper_aperture_target]`, and two 128-unit hidden layers.
- [x] Report one-step RMSE and 0.5-second free-rollout RMSE against both MuJoCo and a persistence predictor, grouped by held-out clip. Keep checkpoints and predictions local.
- [x] Run synthetic tests, then train/evaluate on the real-data-derived transitions. The model slightly improves one-step RMSE but is worse than persistence in a 0.5-second rollout; report that limit without claiming reliable predictive fidelity.
- [x] Commit as `feat: learn action-conditioned task dynamics` (`10e98e3`).

## Task 5: Publish the Application-Ready Story

**Files:** update `README.md`, `docs/data_collection.md`, and aggregate-only `results/metrics/` and `results/figures/`.

- [x] Update the project flow to show collected video → retargeted action → physical contact manipulation → learned state prediction.
- [x] Publish a compact table for robot task success by human label and world-model held-out error versus persistence; clearly state episode counts and simulation-proxy limits.
- [x] Document installation of the optional model environment, local layout JSON format, training/evaluation commands, and the local paired-demo command.
- [x] Run all core tests and optional world-model tests, verify documented CLI commands with `--help`, inspect staged files for personal data, and scan the public tree/commit messages for development-tool references.
- [ ] Commit as `docs: publish contact task and world model results` and push to `main` after final review.
