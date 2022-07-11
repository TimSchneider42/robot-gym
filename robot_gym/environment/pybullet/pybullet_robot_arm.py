from typing import Tuple, Optional

import pybullet

import numpy as np

from .pybullet_robot_component import PybulletRobotComponent
from transformation import Transformation
from robot_gym.environment.generic import RobotArm
from pyboolet.multibody import Multibody, URDFBody
from pyboolet.simulation_object import Link
from robot_gym.environment.simulation import SimulationJoint
from robot_gym import logger
from ..util import cartesian_vel_to_joint_vel


class PyBulletRobotArm(RobotArm, PybulletRobotComponent):
    def __init__(self, wrapped_body: Multibody, joints: Tuple[SimulationJoint, ...], bg_arm: URDFBody,
                 time_step: float, wrapped_root_link: Optional[Link] = None,
                 virtual_root_link_transformation: Optional[Transformation] = None,
                 boundary_link: Optional[Link] = None, linear_target_acceleration_lin: float = 1.2,
                 linear_target_acceleration_ang: float = 1.2):
        PybulletRobotComponent.__init__(self, wrapped_body, joints, wrapped_root_link, virtual_root_link_transformation,
                                        boundary_link)
        RobotArm.__init__(self)
        self.__bg_arm = bg_arm
        self.__linear_target_pose: Optional[Transformation] = None
        self.__linear_target_lin_vel_vec = None
        self.__linear_target_ang_vel_vec = None
        self.__linear_target_lin_vel_scalar = None
        self.__linear_target_ang_vel_scalar = None
        self.linear_acceleration_lin = 1.2
        self.linear_acceleration_ang = 1.2
        self.__linear_ang_vel_max = 2.0
        self.__time_step = time_step
        self.__tcp_link = self.wrapped_body.joints["gripper_wrist_joint"].child

    def solve_ik(self, ee_pose: Transformation) -> np.ndarray:
        # TODO: Use just the arm for IK calculations (without the gripper)
        revolute_joints = [j.wrapped_joint for j in self.joints]
        joint_intervals = np.array([j.interval for j in revolute_joints])
        joint_positions = np.array([j.joint_position for j in revolute_joints])

        limits = joint_intervals
        endeffector_index = self.__tcp_link.link_index
        target_joint_positions = self.wrapped_body.call(
            pybullet.calculateInverseKinematics,
            endEffectorLinkIndex=endeffector_index,
            targetPosition=ee_pose.translation, targetOrientation=ee_pose.quaternion,
            lowerLimits=limits[:, 0], upperLimits=limits[:, 1],
            jointRanges=joint_intervals[:, 1] - joint_intervals[:, 0], restPoses=joint_positions, maxNumIterations=1000)
        return target_joint_positions

    def move_to_pose(self, target_pose: Transformation):
        target_joint_positions = self.solve_ik(target_pose)
        self.set_joint_target_positions(target_joint_positions[:len(self.joints)])
        self.move_to_joint_positions(target_joint_positions[:len(self.joints)])
        error = np.linalg.norm(self.__tcp_link.pose.translation - target_pose.translation)
        if error > 1e-2:
            logger.warning("Error in IK solution is too large ({}).".format(error))

    def move_towards_pose_linear(self, target_pose: Transformation, linear_velocity: float = 0.25,
                                 angular_velocity: Optional[float] = None):
        self.__linear_target_lin_vel_vec = None
        self.__linear_target_pose = target_pose
        self.__linear_target_lin_vel_scalar = linear_velocity
        self.__linear_target_ang_vel_scalar = angular_velocity

    def compute_jacobian(self, joint_positions: np.ndarray) -> np.ndarray:
        linear_jacobian, angular_jacobian = self.__tcp_link.call(
            pybullet.calculateJacobian, localPosition=[0, 0, 0], objPositions=list(joint_positions),
            objVelocities=[0] * len(joint_positions), objAccelerations=[0] * len(joint_positions))
        return np.concatenate([linear_jacobian, angular_jacobian])

    def move_cartesian_velocity(self, linear_vel: np.ndarray, angular_vel: np.ndarray):
        self.__linear_target_pose = None
        self.__linear_target_lin_vel_vec = linear_vel
        self.__linear_target_ang_vel_vec = angular_vel

    def __update_linear_vel(self, linear_vel: np.ndarray, angular_vel: np.ndarray):
        current_vel_lin, current_vel_ang = self.__tcp_link.velocity
        acc_diff_lin = linear_vel - current_vel_lin
        acc_diff_lin_norm = np.linalg.norm(acc_diff_lin)
        effective_diff_lin = min(acc_diff_lin_norm, self.__time_step * self.linear_acceleration_lin)
        scale_lin = 0.0 if acc_diff_lin_norm == 0.0 else effective_diff_lin / acc_diff_lin_norm
        target_vel_lin = current_vel_lin + acc_diff_lin * scale_lin

        acc_diff_ang = angular_vel - current_vel_ang
        acc_diff_ang_norm = np.linalg.norm(acc_diff_ang)
        effective_diff_ang = min(acc_diff_ang_norm, self.__time_step * self.linear_acceleration_ang)
        scale_ang = 0.0 if acc_diff_ang_norm == 0.0 else effective_diff_ang / acc_diff_ang_norm
        target_vel_ang = current_vel_ang + acc_diff_ang * scale_ang

        target_vel_lin_rf, target_vel_ang_rf = self.pose.rotation.apply(
            np.stack([target_vel_lin, target_vel_ang]), inverse=True)

        jacobian = self.compute_jacobian(self.joint_positions)
        joint_velocities = cartesian_vel_to_joint_vel(
            jacobian, np.concatenate([target_vel_lin_rf, target_vel_ang_rf]),
            self.joint_velocity_limits)
        self.set_joint_target_velocities(joint_velocities)

    def update(self):
        if self.__linear_target_pose is not None:
            target_pose = self.__linear_target_pose
            current_pose = self.__tcp_link.pose

            lin_diff = target_pose.translation - current_pose.translation
            lin_error = np.linalg.norm(lin_diff)
            if lin_error > 5e-4:
                lin_diff_norm = lin_diff / (lin_error + 1e-5)
                target_vel_lin_mag = np.sqrt(lin_error)
                target_vel_lin_mag_clipped = min(self.__linear_target_lin_vel_scalar, target_vel_lin_mag)
                target_vel_lin = lin_diff_norm * target_vel_lin_mag_clipped
            else:
                target_vel_lin = np.zeros(3)

            ang_diff = current_pose.rotation.inv() * target_pose.rotation
            ang_rotvec = current_pose.rotation.apply(ang_diff.as_rotvec())
            ang_error = np.linalg.norm(ang_rotvec)
            if ang_error > 5e-4:
                ang_diff_norm = ang_rotvec / (ang_error + 1e-5)
                target_vel_ang_mag = np.sqrt(ang_error)
                if self.__linear_target_ang_vel_scalar is None and lin_error != 0:
                    lin_vel = np.linalg.norm(target_vel_lin)
                    max_ang_vel = min(self.__linear_ang_vel_max, lin_vel * ang_error / lin_error)
                else:
                    max_ang_vel = self.__linear_target_ang_vel_scalar
                target_vel_ang_mag_clipped = min(max_ang_vel, target_vel_ang_mag)
                target_vel_ang = ang_diff_norm * target_vel_ang_mag_clipped
            else:
                target_vel_ang = np.zeros(3)

            self.__update_linear_vel(target_vel_lin, target_vel_ang)
        elif self.__linear_target_lin_vel_vec is not None and self.__linear_target_ang_vel_vec is not None:
            self.__update_linear_vel(self.__linear_target_lin_vel_vec, self.__linear_target_ang_vel_vec)

    def stop_linear_control(self):
        self.__linear_target_pose = self.__linear_target_lin_vel_vec = None
