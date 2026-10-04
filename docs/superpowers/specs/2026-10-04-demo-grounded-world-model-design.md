# Demonstration-Grounded World Model and Research Showcase

## Goal

Turn MORPH's existing action-conditioned state predictor into a defensible learned simulator for the Humanoid internship challenge. The applicant's 16 overhead videos must remain the source of the robot trajectories and the primary evaluation distribution. The repository must show one real demonstration beside its MuJoCo Panda replay and explain what the model predicts, where it is accurate, and where it fails.

## Challenge requirements

The challenge asks an applicant to use personally collected data to drive a manipulator in simulation. It explicitly accepts world-model work that demonstrates video or state prediction, puts less emphasis on policy performance, and evaluates the quality of world-model predictions. The listed world-model directions include action-conditioned prediction over long horizons, offline policy scoring, synthetic rollouts, and model-fidelity metrics.

## Existing system and boundary

MORPH already processes 16 personally recorded demonstrations, maps tracked palm motion into Panda commands, replays each command sequence in a physical-contact MuJoCo scene, and records state/action transitions. A 37-state/4-action predictor has been evaluated with folds grouped by source video. A separate 240-episode randomized simulator dataset and its checkpoint exist; those results are supplementary and must not be mixed with the human-demonstration results.

The world model in this project predicts structured simulator state, not RGB frames. Human video affects the model experiment through the tracked trajectory that drives the Panda simulation; video pixels are not a direct model input. Public results and the README must state this distinction plainly.

## Design

Use only the existing human-demonstration-driven transition archives as the primary world-model benchmark. Keep direct and confidence-aware executions from each source video in the same outer fold. Train and calibrate on the remaining clips, then evaluate on held-out source clips. No transition, trajectory, or derived clip variant from an outer test video may enter fitting, early stopping, gain calibration, or candidate selection.

Treat the fitted dynamics model as a learned simulator. For each held-out video and each available retargeting method, roll the model forward using that method's recorded actions. Derive a predicted task outcome from the terminal package pose, package support/gripper contacts, and the existing physical success definition. Compare this prediction with the corresponding MuJoCo replay outcome. Use the model score to rank candidate method/trajectory outcomes and report ranking quality and regret alongside state-prediction metrics. If existing archives do not preserve enough episode alignment or fields to compute a metric without leakage, report the missing evidence and omit that metric rather than reconstructing it from filenames or human labels.

Measure fidelity at the archived time interval over one-step, 0.5-second, and the longest supported common rollout horizon. Report physical state groups separately, include persistence and appropriate zero-velocity baselines, and retain contact precision/recall/F1. Compare ensemble disagreement against realized rollout error by held-out clip to show where the simulator should not be trusted. Check physical invariants that are observable in the archive, including finite states, unit package quaternions, binary contact flags, and plausible contact-conditioned package motion. Keep per-clip videos, paths, and records private; publish only aggregate metrics, uncertainty/fidelity plots, and the explicitly requested short showcase video.

## Showcase video and README

Select one clear overhead human demonstration for which the corresponding Panda replay can be reproduced. Produce a short side-by-side comparison with the human clip on one side and the Panda simulation on the other, aligned by task phase and labeled as human video versus MuJoCo replay. Keep original recordings and full-resolution processed clips outside Git. Place only a compressed, short derived comparison asset and a lightweight preview in the repository; document that the simulation is a replay driven by the demonstration, not a physical robot recording.

Rewrite the README in a research-project style with: a concise abstract and contribution; the human-to-simulation video comparison near the top; task and data protocol; state/action definition; model and grouped evaluation protocol; held-out metrics and baselines; policy/outcome-ranking and model-trust results when supported; representative failure analysis; reproducibility commands; limitations and future work. Clearly separate human-video-driven model results from the randomized-simulation experiment. Preserve measured negative results. Do not imply a VLA, raw-video generative model, or physical-robot validation.

## Acceptance criteria

1. The primary world-model experiment trains and evaluates on transitions generated by trajectories extracted from the applicant's videos, with strict source-video-held-out evaluation.
2. A learned-simulator score is compared with actual MuJoCo outcomes for held-out demonstration-driven rollouts wherever the archives support an honest comparison.
3. Reports include multi-horizon physical state errors, strong simple baselines, contact metrics, and an uncertainty-versus-error fidelity measure. Any metric not supported by aligned held-out records is explicitly omitted with a reason.
4. Results distinguish human-labeled attempt outcome from simulated robot outcome and do not claim that simulation demonstrates physical robot performance.
5. One real overhead clip and its Panda MuJoCo replay are presented together in a short, readable, compressed video asset; original source videos remain outside Git.
6. The README is reproducible, internally consistent with checked-in metrics, explains design choices and failed results, and presents the video comparison and world-model evidence clearly.
7. All model, dataset, video, checkpoint, and metric artifacts retain provenance sufficient to reproduce the published aggregates. Public artifacts contain no per-clip personal records or local machine paths.

## Constraints

- Preserve the existing 16 human videos and all previously collected local data.
- Keep the randomized 240-episode dataset as a separately labeled supporting experiment.
- Do not train a VLA or claim video-frame prediction as part of this scope.
- Do not tune on outer test clips or present test-selected checkpoints/gains.
- Keep original personal recordings, per-clip outputs, and model checkpoints out of Git.
- Use measured results only; a negative or mixed result is acceptable if reported honestly.
- The role listing gives a submission deadline of 9 October 2026 at 23:59 BST; keep the implementation focused enough to produce a coherent, reproducible submission before then.
