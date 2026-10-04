"""Render a Panda path and assemble it with a tracked human demonstration."""

from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np

from evaluation.demo_evaluator import REPLAY_SEED
from simulation.panda_env import PandaEnv
from simulation.trajectory_player import CartesianTrajectoryPlayer


ROBOT_WIDTH = 480
ROBOT_HEIGHT = 360
PANEL_HEIGHT = 360
HUMAN_WIDTH = 640
HEADER_HEIGHT = 48
OUTPUT_WIDTH = HUMAN_WIDTH + ROBOT_WIDTH
OUTPUT_HEIGHT = PANEL_HEIGHT + HEADER_HEIGHT
BACKGROUND = (19, 22, 26)
PANEL_TITLES = ("Human demonstration", "MuJoCo Panda replay")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_SHOWCASE_PATH = REPOSITORY_ROOT / "results" / "videos" / "human_to_panda.mp4"


def _local_output_path(path: str | Path) -> Path:
    """Resolve an output and reject media paths inside the Git checkout."""
    output = Path(path).expanduser().resolve()
    repository_root = REPOSITORY_ROOT
    if output == repository_root or repository_root in output.parents:
        raise ValueError("real and derived videos must be saved outside the repository")
    return output


def _showcase_output_path(path: str | Path) -> Path:
    """Allow only the one documented derived comparison asset into the repo."""
    output = Path(path).expanduser().resolve()
    if output != PUBLIC_SHOWCASE_PATH.resolve():
        raise ValueError(f"public showcase output is limited to {PUBLIC_SHOWCASE_PATH}")
    return output


def render_panda_trajectory(
    trajectory: np.ndarray,
    output_path: str | Path,
    model_path: str | Path | None = None,
) -> Path:
    """Render a measured Panda replay to a local MP4 with a fixed camera."""
    try:
        points = np.asarray(trajectory, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("trajectory must contain numeric XYZ coordinates") from error
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise ValueError("trajectory must contain at least two XYZ waypoints")
    if not np.isfinite(points).all():
        raise ValueError("trajectory coordinates must all be finite")

    destination = _local_output_path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    env = PandaEnv() if model_path is None else PandaEnv(model_path)
    env.data.qpos[: REPLAY_SEED.size] = REPLAY_SEED
    env.data.ctrl[: REPLAY_SEED.size] = REPLAY_SEED
    mujoco.mj_forward(env.model, env.data)

    try:
        renderer = mujoco.Renderer(env.model, height=ROBOT_HEIGHT, width=ROBOT_WIDTH)
    except Exception as error:
        raise RuntimeError(
            "MuJoCo offscreen rendering could not initialize; use the project's "
            "configured Python environment and verify graphics support"
        ) from error

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = (0.40, 0.0, 0.48)
    camera.distance = 1.65
    camera.azimuth = 135.0
    camera.elevation = -30.0
    fps = 1.0 / 0.02
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ROBOT_WIDTH, ROBOT_HEIGHT)
    )
    if not writer.isOpened():
        renderer.close()
        raise OSError(f"Could not create Panda replay video: {destination}")

    def capture_frame() -> None:
        renderer.update_scene(env.data, camera=camera)
        rgb = renderer.render()
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    try:
        capture_frame()
        player = CartesianTrajectoryPlayer(env)
        player.follow(points, on_control_step=capture_frame)
    except Exception:
        writer.release()
        renderer.close()
        raise
    writer.release()
    renderer.close()
    return destination


def _video_frame_count(capture: cv2.VideoCapture, path: Path) -> int:
    count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if count < 1:
        raise ValueError(f"video contains no readable frames: {path}")
    return count


def _read_to_index(
    capture: cv2.VideoCapture,
    current_index: int,
    target_index: int,
    current_frame: np.ndarray | None,
    path: Path,
) -> tuple[int, np.ndarray]:
    while current_index < target_index:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"could not decode frame {target_index} from '{path}'")
        current_index += 1
        current_frame = frame
    if current_frame is None:
        raise ValueError(f"video contains no readable frames: {path}")
    return current_index, current_frame


def _fit_panel(frame: np.ndarray, width: int) -> np.ndarray:
    panel = np.full((PANEL_HEIGHT, width, 3), BACKGROUND, dtype=np.uint8)
    frame_height, frame_width = frame.shape[:2]
    scale = min(width / frame_width, PANEL_HEIGHT / frame_height)
    fitted_width = max(1, int(round(frame_width * scale)))
    fitted_height = max(1, int(round(frame_height * scale)))
    resized = cv2.resize(frame, (fitted_width, fitted_height), interpolation=cv2.INTER_AREA)
    x = (width - fitted_width) // 2
    y = (PANEL_HEIGHT - fitted_height) // 2
    panel[y : y + fitted_height, x : x + fitted_width] = resized
    return panel


def _draw_title(frame: np.ndarray, text: str, x0: int, width: int, color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (x0, 0), (x0 + width, HEADER_HEIGHT), color, -1)
    font_scale = 0.62
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    x = x0 + max(8, (width - text_size[0]) // 2)
    y = (HEADER_HEIGHT + text_size[1]) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def make_paired_video(
    real_path: str | Path,
    robot_path: str | Path,
    output_path: str | Path,
    fps: float = 30.0,
    *,
    public_showcase: bool = False,
) -> Path:
    """Pair videos by normalized progress; public output requires explicit opt-in."""
    real = Path(real_path).expanduser().resolve()
    robot = Path(robot_path).expanduser().resolve()
    output = _showcase_output_path(output_path) if public_showcase else _local_output_path(output_path)
    if not real.is_file() or not robot.is_file():
        raise FileNotFoundError("both the real demonstration and Panda replay videos must exist")
    if output in {real, robot}:
        raise ValueError("paired video output must not overwrite an input video")
    if output.exists():
        try:
            if output.samefile(real) or output.samefile(robot):
                raise ValueError("paired video output must not overwrite an input video")
        except OSError:
            pass
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")

    real_capture = cv2.VideoCapture(str(real))
    robot_capture = cv2.VideoCapture(str(robot))
    if not real_capture.isOpened() or not robot_capture.isOpened():
        real_capture.release()
        robot_capture.release()
        raise ValueError("could not open both input videos")
    real_count = _video_frame_count(real_capture, real)
    robot_count = _video_frame_count(robot_capture, robot)
    total_frames = max(real_count, robot_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (OUTPUT_WIDTH, OUTPUT_HEIGHT)
    )
    if not writer.isOpened():
        real_capture.release()
        robot_capture.release()
        raise OSError(f"Could not create paired video: {output}")

    real_index = robot_index = -1
    real_frame = robot_frame = None
    try:
        for output_index in range(total_frames):
            progress = output_index / max(1, total_frames - 1)
            target_real = int(round(progress * (real_count - 1)))
            target_robot = int(round(progress * (robot_count - 1)))
            real_index, real_frame = _read_to_index(real_capture, real_index, target_real, real_frame, real)
            robot_index, robot_frame = _read_to_index(robot_capture, robot_index, target_robot, robot_frame, robot)
            human_panel = _fit_panel(real_frame, HUMAN_WIDTH)
            robot_panel = _fit_panel(robot_frame, ROBOT_WIDTH)
            canvas = np.full((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), BACKGROUND, dtype=np.uint8)
            canvas[HEADER_HEIGHT:, :HUMAN_WIDTH] = human_panel
            canvas[HEADER_HEIGHT:, HUMAN_WIDTH:] = robot_panel
            _draw_title(canvas, PANEL_TITLES[0], 0, HUMAN_WIDTH, (64, 88, 107))
            _draw_title(canvas, PANEL_TITLES[1], HUMAN_WIDTH, ROBOT_WIDTH, (55, 102, 89))
            writer.write(canvas)
    finally:
        real_capture.release()
        robot_capture.release()
        writer.release()
    return output
