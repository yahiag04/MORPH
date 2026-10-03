# Human Demonstration Collection Protocol

## Task and labels

The current dataset contains 16 overhead recordings of a person moving a small boxed item toward a bowl/target area: 12 human-labeled successful attempts (`riusciti`) and 4 failed attempts (`falliti`). These labels describe only what happened in the video. A retargeted hand path guides a Panda controller in MuJoCo, where a package is picked up and placed through physical contact with a tray. The recordings do not contain robot actions or robot state labels.

For a comparable batch, keep the item, start area, target area, and workspace layout consistent. Record complete attempts with a brief pause before and after each. Include unsuccessful attempts rather than silently replacing them. Put video files one directory level under label folders, such as `riusciti/` and `falliti/`.

## Camera and workspace

- Fix the phone above the table in a top-down view; do not move it between demonstrations.
- Show a rectangular workspace with four visible pink corner markers. Keep the full hand, item path, and target visible for the whole recording.
- Use steady lighting and a plain, contrasting table surface. Avoid faces and unrelated people.
- Record at 720p or higher and at least 30 frames per second. Use original MP4/MOV clips without trimming or stabilization edits.
- Keep the item, start area, target area, and workspace layout consistent across runs.

## Import and local storage

Copy the phone's original clips to the computer and place them in labeled subfolders. The batch processor accepts MP4, MOV, M4V, and AVI files. Original videos, per-frame tracking CSVs, calibration JSON, review images, and per-clip evaluation files must remain outside the repository.

## Processing and replay

```bash
source ~/Documents/.venv/bin/activate
PYTHONPATH=src python scripts/process_dataset.py /path/to/videos \
  --output-dir /path/to/local_morph_data
PYTHONPATH=src python scripts/calibrate_workspace.py /path/to/videos \
  --output-dir /path/to/local_morph_data/calibrations
PYTHONPATH=src python scripts/evaluate_demonstrations.py \
  --processing-manifest /path/to/local_morph_data/processing_manifest.json \
  --calibration-dir /path/to/local_morph_data/calibrations \
  --output-dir /path/to/local_morph_evaluation
PYTHONPATH=src python scripts/evaluate_contact_demos.py \
  --processing-manifest /path/to/local_morph_data/processing_manifest.json \
  --calibration-dir /path/to/local_morph_data/calibrations \
  --output-dir /path/to/local_contact_evaluation
```

The first processing run downloads Google's version 1 MediaPipe hand-landmarker model to the ignored `assets/models/` directory and verifies its SHA-256 checksum. Frame processing stays on this computer; the MediaPipe package reports API usage and performance metrics to Google.

The calibration detector orders the four pink markers as top-left, top-right, bottom-right, and bottom-left. Inspect the generated marker review images for every clip. The corners map to robot x=0.38–0.68 m and y=−0.22–0.22 m at fixed z=0.62 m. These robot bounds are assumptions, not measured camera-to-robot geometry. Image y is inverted for robot y. Monocular video provides no metric hand height, so no physical Z is inferred.

Low-confidence and missing points are interpolated only if they occupy no more than 20% of frames; mapped motion is smoothed and capped at 0.35 m/s. Inspect the annotated clip and robot workspace before interpreting a trajectory. Contact evaluation detects the package and bowl location in each video when possible; per-clip results and transition archives stay local. Only aggregates are checked in. The package state advances through MuJoCo contacts, without attaching or moving it in software. To create a side-by-side human/Panda MP4, pass an annotated clip and its trajectory CSV to `scripts/make_paired_demo.py`; keep those videos local.

If automatic package/bowl detection is unavailable, pass `--task-layout /path/to/layout.json` as a fallback. Coordinates are normalized to the rectified workspace, with `(0, 0)` at its top-left and `(1, 1)` at its bottom-right:

```json
{
  "pickup_uv": [0.24, 0.31],
  "dropoff_uv": [0.73, 0.68]
}
```

## Known limitations

The tracker reports a hand-classification score as a confidence proxy, not a calibrated probability of localization accuracy. It follows the nearest palm and marks jumps greater than 0.25 normalized image-coordinate units as missing. This reduces large identity jumps but does not guarantee hand identity when palms are close or cross.

The contact replay succeeded on 5/16 direct-retargeted episodes and 4/16 confidence-aware episodes. These are simulated outcomes, and confidence-aware filtering did not materially change success on this batch. The learned state model improves one-step RMSE slightly over persistence on held-out clips but performs worse over a 0.5-second rollout. Neither result measures physical robot accuracy; validate camera geometry, control, and contact on hardware before making real-world claims.
