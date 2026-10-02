"""Cartesian waypoint interpolation and smoothing utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import mujoco
import numpy as np

from .ik import ARM_JOINT_NAMES, IKResult, solve_position_ik
from .panda_env import PandaEnv


def _positive_finite_scalar(name: str, value: float) -> float:
    """Validate and normalize a positive finite numeric scalar."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite positive number")
    if not np.isrealobj(value):
        raise ValueError(f"{name} must be a finite positive real number")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric_value


def _waypoint_array(waypoints: np.ndarray) -> np.ndarray:
    """Return validated, copied XYZ waypoints with repeated segments removed."""
    try:
        points = np.array(waypoints, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("waypoints must contain numeric XYZ coordinates") from error
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("waypoints must have shape (N, 3)")
    if points.shape[0] < 2:
        raise ValueError("waypoints must contain at least two points")
    if not np.isfinite(points).all():
        raise ValueError("waypoint coordinates must all be finite")

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 0.0))
    return points[keep]


def resample_waypoints(
    waypoints: np.ndarray,
    speed: float,
    sample_period: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample an XYZ path at constant arc-length speed.

    Consecutive duplicate points are ignored. The returned timestamps start at
    zero and include the path endpoint, even when its time is not a multiple of
    ``sample_period``.

    Args:
        waypoints: Finite array with shape ``(N, 3)`` and at least two points.
        speed: Positive Cartesian path speed in meters per second.
        sample_period: Positive time between regular samples in seconds.

    Returns:
        A pair ``(timestamps, xyz_samples)`` with the final endpoint included.

    Raises:
        ValueError: If the path, speed, or sample period is invalid.
    """
    points = _waypoint_array(waypoints)
    speed = _positive_finite_scalar("speed", speed)
    sample_period = _positive_finite_scalar("sample_period", sample_period)

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_distance = float(segment_lengths.sum())
    if not np.isfinite(total_distance) or total_distance <= 0.0:
        raise ValueError("waypoints must define a nonzero path distance")

    duration = total_distance / speed
    timestamps = np.arange(0.0, duration, sample_period, dtype=np.float64)
    if timestamps.size == 0:
        timestamps = np.array([0.0], dtype=np.float64)
    if duration - timestamps[-1] > 1e-12:
        timestamps = np.append(timestamps, duration)
    else:
        timestamps[-1] = duration

    distances = np.minimum(speed * timestamps, total_distance)
    cumulative_distance = np.concatenate(
        ([0.0], np.cumsum(segment_lengths, dtype=np.float64))
    )
    xyz_samples = np.column_stack(
        [np.interp(distances, cumulative_distance, points[:, axis]) for axis in range(3)]
    )
    xyz_samples[0] = points[0]
    xyz_samples[-1] = points[-1]
    return timestamps, xyz_samples


def smooth_trajectory(samples: np.ndarray, window_size: int) -> np.ndarray:
    """Smooth interior XYZ samples with a moving average and fixed endpoints.

    When ``window_size`` exceeds the available samples, the effective window
    is capped to the largest usable odd size. A size of one returns a copy.
    """
    try:
        points = np.array(samples, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("samples must contain numeric XYZ coordinates") from error
    if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] < 1:
        raise ValueError("samples must have shape (N, 3) with at least one row")
    if not np.isfinite(points).all():
        raise ValueError("sample coordinates must all be finite")
    if (
        isinstance(window_size, bool)
        or not isinstance(window_size, (int, np.integer))
        or window_size < 1
        or window_size % 2 == 0
    ):
        raise ValueError("window_size must be a positive odd integer")

    largest_odd_window = points.shape[0] if points.shape[0] % 2 else points.shape[0] - 1
    effective_window = min(int(window_size), max(1, largest_odd_window))
    if effective_window == 1 or points.shape[0] < 3:
        return points

    radius = effective_window // 2
    padded = np.pad(points, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.full(effective_window, 1.0 / effective_window)
    smoothed = np.column_stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(3)]
    )
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed


@dataclass(frozen=True)
class TrajectoryLog:
    """Samples recorded while following a Cartesian trajectory."""

    timestamps: np.ndarray
    target_xyz: np.ndarray
    actual_xyz: np.ndarray
    ik_converged: np.ndarray

    @property
    def position_error(self) -> np.ndarray:
        """Return per-sample Euclidean hand-position error in meters."""
        return np.linalg.norm(self.target_xyz - self.actual_xyz, axis=1)

    def write_csv(self, output_path: str | Path) -> Path:
        """Write target, actual, and IK status columns to ``output_path``."""
        path = Path(output_path).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "timestamp",
                        "target_x",
                        "target_y",
                        "target_z",
                        "actual_x",
                        "actual_y",
                        "actual_z",
                        "ik_converged",
                    ]
                )
                for timestamp, target, actual, converged in zip(
                    self.timestamps,
                    self.target_xyz,
                    self.actual_xyz,
                    self.ik_converged,
                    strict=True,
                ):
                    writer.writerow(
                        [
                            f"{timestamp:.9f}",
                            *(f"{coordinate:.9f}" for coordinate in target),
                            *(f"{coordinate:.9f}" for coordinate in actual),
                            bool(converged),
                        ]
                    )
        except OSError as error:
            raise OSError(f"Could not write trajectory CSV to '{path}': {error}") from error
        return path


class CartesianTrajectoryPlayer:
    """Follow Cartesian samples using Panda IK and its joint position actuators."""

    def __init__(
        self,
        env: PandaEnv,
        speed: float = 0.08,
        control_period: float = 0.02,
        smoothing_window: int = 5,
    ) -> None:
        """Create a player for a loaded Panda simulation."""
        self.env = env
        self.speed = _positive_finite_scalar("speed", speed)
        self.control_period = _positive_finite_scalar(
            "control_period", control_period
        )
        if (
            isinstance(smoothing_window, bool)
            or not isinstance(smoothing_window, (int, np.integer))
            or smoothing_window < 1
            or smoothing_window % 2 == 0
        ):
            raise ValueError("smoothing_window must be a positive odd integer")
        self.smoothing_window = int(smoothing_window)

        simulation_timestep = float(env.model.opt.timestep)
        steps = self.control_period / simulation_timestep
        rounded_steps = int(round(steps))
        if (
            rounded_steps < 1
            or not np.isclose(steps, rounded_steps, rtol=1e-8, atol=1e-9)
        ):
            raise ValueError(
                "control_period must be an integer multiple of the MuJoCo timestep"
            )
        self.steps_per_control = rounded_steps
        self.simulation_timestep = simulation_timestep
        self.arm_actuator_ids = self._resolve_arm_actuators()

    def _resolve_arm_actuators(self) -> np.ndarray:
        """Resolve and validate position actuators matching Panda arm joints."""
        actuator_ids: list[int] = []
        for joint_name in ARM_JOINT_NAMES:
            actuator_name = f"actuator{int(joint_name.removeprefix('joint'))}"
            actuator_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            joint_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if actuator_id < 0:
                raise ValueError(f"Panda model is missing actuator '{actuator_name}'")
            if joint_id < 0:
                raise ValueError(
                    f"Panda model is missing joint '{joint_name}' for '{actuator_name}'"
                )
            if (
                self.env.model.actuator_trntype[actuator_id]
                != mujoco.mjtTrn.mjTRN_JOINT
                or self.env.model.actuator_trnid[actuator_id, 0] != joint_id
            ):
                raise ValueError(
                    f"'{actuator_name}' must actuate Panda joint '{joint_name}'"
                )
            if (
                self.env.model.actuator_gaintype[actuator_id]
                != mujoco.mjtGain.mjGAIN_FIXED
                or self.env.model.actuator_biastype[actuator_id]
                != mujoco.mjtBias.mjBIAS_AFFINE
                or not np.isclose(
                    self.env.model.actuator_gainprm[actuator_id, 0],
                    -self.env.model.actuator_biasprm[actuator_id, 1],
                )
            ):
                raise ValueError(
                    f"'{actuator_name}' must be a position actuator for '{joint_name}'"
                )
            actuator_ids.append(actuator_id)
        return np.asarray(actuator_ids, dtype=np.int32)

    def follow(
        self,
        waypoints: np.ndarray,
        output_csv: str | Path | None = None,
        on_control_step: Callable[[], None] | None = None,
    ) -> TrajectoryLog:
        """Run a Cartesian path and return per-simulation-step measurements.

        Args:
            waypoints: Finite XYZ path in world coordinates.
            output_csv: Optional path for a CSV log.
            on_control_step: Optional callback after each control interval.

        Returns:
            The recorded simulation timestamps, targets, actual XYZ, and IK
            convergence flags.
        """
        sample_times, samples = resample_waypoints(
            waypoints, self.speed, self.control_period
        )
        targets = smooth_trajectory(samples, self.smoothing_window)
        timestamps: list[float] = []
        target_rows: list[np.ndarray] = []
        actual_rows: list[np.ndarray] = []
        convergence_rows: list[bool] = []

        for index, target in enumerate(targets):
            qpos_before_ik = self.env.data.qpos.copy()
            result: IKResult = solve_position_ik(self.env, target)
            joint_target = result.q.copy()
            self.env.data.qpos[:] = qpos_before_ik
            mujoco.mj_forward(self.env.model, self.env.data)
            self.env.data.ctrl[self.arm_actuator_ids] = joint_target

            if index + 1 < sample_times.size:
                interval = sample_times[index + 1] - sample_times[index]
                step_count = max(
                    1, int(round(interval / self.simulation_timestep))
                )
            else:
                step_count = self.steps_per_control

            for _ in range(step_count):
                self.env.step()
                timestamps.append(float(self.env.data.time))
                target_rows.append(target.copy())
                actual_rows.append(self.env.get_end_effector_position())
                convergence_rows.append(result.converged)

            if on_control_step is not None:
                on_control_step()

        log = TrajectoryLog(
            timestamps=np.asarray(timestamps, dtype=np.float64),
            target_xyz=np.asarray(target_rows, dtype=np.float64),
            actual_xyz=np.asarray(actual_rows, dtype=np.float64),
            ik_converged=np.asarray(convergence_rows, dtype=np.bool_),
        )
        if output_csv is not None:
            log.write_csv(output_csv)
        return log
