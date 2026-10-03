# World Model V2 Implementation Plan

**Goal:** Improve and fairly evaluate multi-step package manipulation dynamics on unseen demonstration clips.

**Architecture:** Extend simulation state to include velocities, package orientation, contacts, and tray context. Train an ensemble with continuous dynamics losses, binary contact classification, and multi-step sequence supervision; evaluate by nested clip-group folds with separated state-group metrics.

**Spec:** `docs/designs/world-model-v2.md`

## Constraints

- Core requirements remain PyTorch-free; training remains in the optional PyTorch environment.
- Raw video, clip identifiers, checkpoints, and per-clip predictions remain outside Git.
- Both retargeting methods for one source clip stay in the same train, validation, or test partition.
- V1 results remain archived and are not compared numerically as though the state schemas were identical.

## Task 1: Log the complete task state

**Files:** `src/evaluation/contact_demo_evaluator.py`, `tests/test_contact_demo_evaluator.py`.

- [x] Add tests for the ordered 37-value state, finite values, and inclusion of Panda velocities, package orientation/angular velocity, gripper velocity, and tray center.
- [x] Expose one canonical state vector function and use it in transition archives.
- [x] Verify reset, deterministic replay, archive shapes, and existing contact outcome tests.
- [x] Commit `feat: record complete contact task state`.

## Task 2: Train multi-step dynamics with discrete contacts

**Files:** `src/world_model/state_dynamics.py`, `tests/test_state_dynamics.py`.

- [x] Add failing tests for state/action dimensions, normalized quaternion rollouts, contact probabilities in `[0, 1]`, correct grouped losses, multi-step prediction, and deterministic ensemble shape.
- [x] Implement continuous delta regression plus binary contact logits, with contact BCE and 1/5/25-step rollout losses from contiguous episodes.
- [x] Fit all normalization on training clips only; select checkpoints using validation rollout loss and calibrate group residual gains on inner validation rollouts.
- [x] Train a five-member ensemble on clip-bootstrap samples; average continuous predictions and contact probabilities during evaluation.
- [x] Verify synthetic known dynamics, no clip leakage, and bounded/contact-valid outputs.
- [ ] Commit `feat: train multi-step contact dynamics ensemble`.

## Task 3: Evaluate on held-out clips

**Files:** `scripts/train_world_model.py`, `src/world_model/state_dynamics.py`, `tests/test_state_dynamics.py`.

- [x] Add nested grouped fold evaluation; source clips never cross partitions, including direct/confidence variants.
- [x] Regenerate V2 transitions from the local 16-video manifest and calibrations; preserve local-only data.
- [x] Run four-fold out-of-fold evaluation with inner grouped validation and state-group metrics for position, velocity, aperture, and contacts.
- [x] Compare each state group with persistence and report contact accuracy/F1, rollout counts, fold spread, and ensemble disagreement.
- [x] Train each outer fold with a deterministic seed; retain checkpoints and detailed metrics locally.
- [ ] Commit `feat: evaluate world model by held-out clip`.

## Task 4: Publish the verified result

**Files:** `README.md`, `docs/data_collection.md`, `results/metrics/world_model_v2.json`, `results/figures/world_model_v2.png`.

- [x] Publish V2 separately from the V1 score, with episode/fold counts, grouped metrics, uncertainty, and simulation limits.
- [x] V2 improves package and robot position over persistence at 0.5 seconds; report the remaining package linear velocity weakness explicitly.
- [x] Run core and optional model tests, CLI help, inspect public files for personal data and development-tool references.
- [ ] Commit `docs: publish world model v2 results` and push to `main`.
