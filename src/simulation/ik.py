"""Position-only inverse kinematics for the Franka Panda in MuJoCo."""

from dataclasses import dataclass

import mujoco
import numpy as np

from .panda_env import PandaEnv


ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))


def _finite_scalar(name: str, value: float) -> float:
    """Return a real finite scalar or raise a parameter-specific error."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite numeric value")
    if not np.isrealobj(value):
        raise ValueError(f"{name} must be a finite real value")
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


@dataclass(frozen=True)
class IKResult:
    """Result of a position inverse-kinematics solve.

    Attributes:
        q: Seven Panda arm joint positions in joint1-to-joint7 order.
        converged: Whether the final hand position is within tolerance.
        position_error: Euclidean XYZ error in meters.
        iterations: Number of joint updates performed.
    """

    q: np.ndarray
    converged: bool
    position_error: float
    iterations: int


def solve_position_ik(
    env: PandaEnv,
    target_xyz: np.ndarray,
    *,
    tolerance: float = 0.015,
    max_iterations: int = 200,
    damping: float = 0.05,
    step_size: float = 0.5,
) -> IKResult:
    """Move the current Panda arm configuration toward a world-frame XYZ.

    The solver changes only the seven arm joint positions in ``env.data``.
    Each position is clipped to its MuJoCo joint range, and the final forward
    kinematics are left available through ``env.get_end_effector_position``.

    Args:
        env: Loaded Panda simulation environment.
        target_xyz: Three finite world-frame coordinates in meters.
        tolerance: Convergence threshold in meters.
        max_iterations: Maximum number of joint updates.
        damping: Positive damping coefficient for damped least squares.
        step_size: Update scale in the interval ``(0, 1]``.

    Returns:
        Joint solution, final error, update count, and convergence status.

    Raises:
        ValueError: If the target, solver parameters, or Panda joint metadata
            are invalid.
    """
    try:
        target = np.asarray(target_xyz, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("target_xyz must contain numeric coordinates") from error
    if target.shape != (3,):
        raise ValueError("target_xyz must contain exactly three coordinates")
    if not np.isfinite(target).all():
        raise ValueError("target_xyz coordinates must all be finite")
    tolerance = _finite_scalar("tolerance", tolerance)
    damping = _finite_scalar("damping", damping)
    step_size = _finite_scalar("step_size", step_size)
    if tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if damping <= 0.0:
        raise ValueError("damping must be finite and positive")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer")
    if not 0.0 < step_size <= 1.0:
        raise ValueError("step_size must be finite and in (0, 1]")

    joint_ids: list[int] = []
    for joint_name in ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0:
            raise ValueError(f"Panda model is missing arm joint '{joint_name}'")
        if env.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Panda arm joint '{joint_name}' must be a hinge")
        if not env.model.jnt_limited[joint_id]:
            raise ValueError(f"Panda arm joint '{joint_name}' must have joint limits")
        joint_ids.append(joint_id)

    qpos_addresses = env.model.jnt_qposadr[joint_ids]
    dof_addresses = env.model.jnt_dofadr[joint_ids]
    limits = env.model.jnt_range[joint_ids]
    if not np.isfinite(limits).all() or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("Panda arm joint limits must be finite and ordered")

    q = np.clip(env.data.qpos[qpos_addresses].copy(), limits[:, 0], limits[:, 1])
    env.data.qpos[qpos_addresses] = q
    jacobian_position = np.zeros((3, env.model.nv), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    iterations = 0

    mujoco.mj_forward(env.model, env.data)
    for _ in range(int(max_iterations)):
        position_error_vector = target - env.data.xpos[env.hand_id]
        if np.linalg.norm(position_error_vector) <= tolerance:
            break

        mujoco.mj_jacBody(
            env.model,
            env.data,
            jacobian_position,
            None,
            env.hand_id,
        )
        jacobian = jacobian_position[:, dof_addresses]
        normal_matrix = jacobian @ jacobian.T + damping**2 * identity
        delta_q = jacobian.T @ np.linalg.solve(
            normal_matrix, position_error_vector
        )

        q = np.clip(
            q + step_size * delta_q,
            limits[:, 0],
            limits[:, 1],
        )
        env.data.qpos[qpos_addresses] = q
        iterations += 1
        mujoco.mj_forward(env.model, env.data)

    mujoco.mj_forward(env.model, env.data)
    final_error = float(
        np.linalg.norm(target - env.data.xpos[env.hand_id])
    )
    return IKResult(
        q=q.copy(),
        converged=final_error <= tolerance,
        position_error=final_error,
        iterations=iterations,
    )
