# Human Demonstration Collection Protocol

## Task

Record a natural hand pick-and-place of the same 5 cm cube used in the Panda scene. Start the cube in one marked square, lift it, carry it across the workspace, and release it in a second marked square. Keep the full hand and both regions visible for the entire recording.

## Camera and workspace

- Use a phone fixed above the table in a top-down view; do not move the camera between demonstrations.
- Show a rectangular workspace with four visible corner marks. These marks define the image bounds used for robot calibration.
- Use steady lighting and a plain, contrasting table surface. Keep the entire object path and hand visible; avoid faces and unrelated people in the frame.
- Record at 720p or higher and at least 30 frames per second. Use original MP4/MOV clips without trimming or stabilization edits.
- Use the same cube, start square, target square, and workspace layout for every run.

## Demonstrations

Record approximately ten complete attempts. Leave a brief pause before and after each attempt, and include unsuccessful grasps rather than silently replacing them. Note any unusual condition in a paper log so it can be entered alongside the video if needed.

## Import and local storage

Copy the phone's original clip to the computer, then import it:

```bash
source ~/Documents/.venv/bin/activate
PYTHONPATH=src python scripts/import_demonstration.py /path/to/phone_clip.mov
```

The importer keeps the original bytes under `data/raw/` and writes a JSON sidecar containing dimensions, frame rate, frame count, and import time. Raw videos and derived per-frame trajectories are excluded from Git by default; they stay on this computer unless you explicitly choose to share them.

For processing, use one clip per file. The camera must remain fixed because calibration maps the four workspace corners to robot XY. Monocular image data has no metric hand height; the baseline therefore uses a fixed robot Z documented in the retargeting configuration.

## Processing and replay

Run the hand tracker on an imported local clip:

```bash
PYTHONPATH=src python scripts/process_video.py data/raw/demo_001.mp4
```

The first run downloads Google's version 1 MediaPipe hand-landmarker task model to the ignored `assets/models/` directory and verifies its SHA-256 checksum. Frame processing stays on this computer; the MediaPipe package reports API usage and performance metrics to Google. Frames and the annotated MP4 remain local under `data/processed/`. To map the observed normalized palm coordinates into Panda coordinates and replay them:

```bash
PYTHONPATH=src python scripts/retarget_trajectory.py data/processed/demo_001.csv --output data/trajectories/demo_001.csv
PYTHONPATH=src mjpython scripts/replay_demo.py data/trajectories/demo_001.csv --results-name demo_001
```

The default image rectangle is normalized x=0.20–0.80 and y=0.15–0.85, mapped to robot x=0.38–0.68 m and y=−0.22–0.22 m at fixed z=0.62 m. These are explicit starter assumptions, not a camera calibration measurement: measure the four workspace corners in the image and set `--image-bounds LEFT RIGHT TOP BOTTOM` and `--robot-bounds X_MIN X_MAX Y_MIN Y_MAX` before interpreting robot replay. Set the fixed plane with `--robot-z`. Image y is inverted for robot y. Low-confidence and missing points are interpolated only if they occupy no more than 20% of frames; mapped motion is smoothed and capped at 0.35 m/s. Check the generated annotated clip and robot workspace before using any trajectory for analysis. Replay saves simulation measurements as CSV, JSON, and PNG under `results/`.

## Known limitation

The tracker reports a hand-classification score as a confidence proxy. It is not a calibrated probability of localization accuracy. Across frames it follows the nearest palm and marks jumps greater than 0.25 normalized image-coordinate units as missing. This reduces large identity jumps, but does not guarantee hand identity when palms are close or cross. The overhead view is intended to make hand XY reliable; no physical Z is inferred from the landmark model's relative depth.
