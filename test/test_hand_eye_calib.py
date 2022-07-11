import unittest
from typing import Literal

import numpy as np

from transformation import Transformation
from robot_gym.environment.real.hand_eye_calibration import estimate_hand_eye_calibration, \
    compute_pose_error


def generate_random_pose(generator, trange, rrange):
    translation_direction = generator.random(3)
    translation_length = generator.random(1) * trange
    translation = translation_direction * translation_length
    rotation_direction = generator.random(3)
    rotation_amount = generator.random(1) * rrange
    rotation_rotvec = rotation_direction * rotation_amount
    pose = Transformation.from_pos_rotvec(translation, rotation_rotvec)
    return pose


class TestTransitionEstimation(unittest.TestCase):
    def test_accuracy(self, backend: Literal["torch", "scipy"] = "torch", places: int = 7):
        generator = np.random.default_rng(0)

        tcp_poses_robot_frame = [generate_random_pose(generator, 10.0, 10.0) for _ in range(20)]

        marker_pose_tcp_frame = generate_random_pose(generator, 0.1, 0.1)
        robot_pose_world_frame = generate_random_pose(generator, 1.0, 1.0)

        marker_poses_world_frame = [robot_pose_world_frame * pose * marker_pose_tcp_frame for pose in
                                    tcp_poses_robot_frame]

        result = estimate_hand_eye_calibration(tcp_poses_robot_frame, marker_poses_world_frame,
                                               rotation_estimation_backend=backend)

        position_error_robot, rot_error_angle_robot = compute_pose_error(
            result.robot_pose_world_frame, robot_pose_world_frame)
        position_error_tcp, rot_error_angle_tcp = compute_pose_error(
            result.marker_pose_tcp_frame, marker_pose_tcp_frame)

        self.assertAlmostEqual(position_error_robot, 0, places=places)
        self.assertAlmostEqual(position_error_tcp, 0, places=places)
        self.assertAlmostEqual(rot_error_angle_robot, 0, places=places)
        self.assertAlmostEqual(rot_error_angle_tcp, 0, places=places)

    def test_accuracy_scipy(self):
        self.test_accuracy(backend="scipy", places=4)
