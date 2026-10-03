# Contact data expansion

**Goal:** Collect complete, timed manipulation transitions and an initial 240-episode synthetic dataset, without changing the existing controller merely to sample more often.

**Design:** Keep the current 100 ms controller interval and observe every 20 ms. Capture each observation, including settling, with actual simulation start/end time and all eight actuator commands. Keep the 37-state and four high-level action schema; store low-level controls as additional fields. Preserve source, episode, scenario family and split provenance. Use 60 seeded scenario families, four interventions per family, and allocate 70/15/15 percent of families to train/validation/test before simulation. Vary layouts, package yaw, speed, pickup estimation error and release behavior; keep geometry, mass and friction fixed for this initial dataset.

**Constraints:** Data and checkpoints stay outside Git. Synthetic episodes never count as human demonstrations. All variants of a family stay in one partition. Record outcomes as observed, including failures; do not label an intended failure as a measured failure. Record sampling interval in archives and reject missing or inconsistent timing in training unless a legacy interval is explicitly supplied. No model retraining in this step.

## Tasks

- [x] Add an observation callback independent of control updates; test that sampling does not change simulator trajectories and captures settling at uniform intervals.
- [x] Export actual times, controls and provenance in each archive; test shapes, continuity, timing and unsupported observation intervals.
- [x] Add deterministic scenario generation and a local-only collection CLI; test grouping, interventions, seed repeatability and overwrite protection.
- [x] Run a smoke collection, then 240 episodes; audit finite states, quaternion norms, controls, phase coverage, timing and partition separation.
- [x] Remove hardcoded model horizon seconds, correct the V2 timing report and chart units, document collection and measured dataset coverage.
- [x] Run relevant regression tests, review the diff, commit and push under the existing authorization.

## Review focus

Sampling must not change control hold time or gripper behavior. Settling must hold the last actuator controls. Invalid or nondividing sampling intervals fail before simulation. Failed collection attempts remain in the manifest. Scenario families cannot cross data partitions. Legacy data cannot silently inherit a 20 ms interval. Raw paths and per-episode identifiers stay out of public summaries.
