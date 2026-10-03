# World Model V2

## Goal

Improve multi-step prediction for the MuJoCo contact task and report whether the learned model beats persistence on unseen video clips. Preserve honest, reproducible metrics; do not optimize or present a score using the final held-out clips.

## Findings from V1

The V1 19-value state omitted Panda joint velocities, package orientation/angular velocity, gripper velocity, and tray location. These variables affect the next state, especially around impacts and tray contact. V1's pooled RMSE combined meters, radians, meters per second, and binary flags, hiding that position rollouts improved while package velocity and contact predictions degraded.

## Design

Use a 37-value state in this exact order: `[ee_xyz(3), arm_qpos(7), arm_qvel(7), package_xyz(3), package_quat_wxyz(4), package_linear_velocity(3), package_angular_velocity(3), gripper_aperture(1), gripper_velocity(1), package_gripper_contact(1), package_support_contact(1), tray_center_xyz(3)]`. The 4-value action remains `[cartesian_target_xyz(3), gripper_aperture_target(1)]`.

Train continuous state deltas with training-only normalization and two contact logits with binary cross entropy. Renormalize predicted package quaternions and threshold contact probabilities during free rollouts. Train on contiguous episode windows with losses at one, five, and 25 control steps so the objective includes the reported 0.5-second horizon. Use an ensemble of independently initialized models to reduce prediction variance and report ensemble disagreement as an uncertainty measure.

Use nested clip-group evaluation: for each outer test fold, hold out complete video clips; split the remaining clips into training and validation groups for early stopping. Direct and confidence-aware replays of the same source clip always stay in the same fold. Publish out-of-fold metrics by physical state group, with both learned model and persistence values, plus contact accuracy/F1. Preserve V1 aggregate metrics so the schema change is explicit.

RTC's action-prefix freezing and inpainting method addresses asynchronous execution of diffusion/flow action chunks. It does not directly fit this low-dimensional state-transition predictor, which receives already specified actions. The transferable continuity idea is applied through contiguous sequence training and autoregressive rollout losses; no RTC policy mechanism is claimed or implemented here. See [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339).

V2 validation selects one residual gain per continuous state group to blend learned deltas with persistence during rollout. Gains are selected only on each fold's inner validation clips; the outer test clips never tune them.

## Limits

The model predicts this MuJoCo task, not real-robot dynamics. The 16 source videos remain the only human demonstrations. More varied data may still be needed; do not add synthetic data to the human demonstration count. Any randomized simulation data must be identified separately and its task parameters included in the model state.
