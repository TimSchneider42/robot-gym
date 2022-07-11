from typing import Optional

import numpy as np


def cartesian_vel_to_joint_vel(
        jacobian: np.ndarray, target_vel: np.ndarray, joint_velocity_limits: np.ndarray,
        current_velocity: Optional[np.ndarray] = None, smoothness_penalty_weight: float = 0.0) -> np.ndarray:
    if smoothness_penalty_weight != 0:
        assert current_velocity is not None, "Current velocity is required to compute smoothness penalty."
        right_side = np.concatenate(
            [target_vel, smoothness_penalty_weight * current_velocity])
        mat = np.concatenate(
            [jacobian, smoothness_penalty_weight * np.eye(current_velocity.shape[0])])
    else:
        right_side = target_vel
        mat = jacobian
    joint_target_velocities = np.linalg.lstsq(mat, right_side, rcond=None)[0]
    relative_velocities = joint_target_velocities / joint_velocity_limits
    scaling = 1 / np.maximum(np.max(np.abs(relative_velocities)), 1)
    scaled_velocities = joint_target_velocities * scaling
    return scaled_velocities
