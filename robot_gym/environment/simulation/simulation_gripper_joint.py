from abc import abstractmethod, ABC
from typing import Sequence, Tuple, Generic, TypeVar

import numpy as np

from robot_gym.environment.simulation import SimulationJoint

from robot_gym.environment.generic import JointMode

JointType = TypeVar("JointType", bound=SimulationJoint)


class SimulationGripperJoint(SimulationJoint, ABC, Generic[JointType]):
    @abstractmethod
    def get_gripper_joints(self) -> Sequence[JointType]:
        pass

    @property
    def gripper_joints(self) -> Sequence[JointType]:
        return self.get_gripper_joints()

    def __to_joint_positions(self, value: float) -> np.ndarray:
        limits_lower, limits_upper = self.__joint_limits()
        return limits_lower + value * (limits_upper - limits_lower)

    def __to_gripper_position(self, joint_positions: np.ndarray) -> float:
        limits_lower, limits_upper = self.__joint_limits()
        if len(limits_lower) == 0:
            return 0.0
        return np.mean((joint_positions - limits_lower) / (limits_upper - limits_lower)).item()

    def __joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        limits = np.array([j.get_joint_interval() for j in self.gripper_joints])
        if len(limits) > 0:
            return limits[:, 0], limits[:, 1]
        else:
            return np.empty((0,)), np.empty((0,))

    def set_joint_position(self, value: float):
        for j, p in zip(self.gripper_joints, self.__to_joint_positions(value)):
            j.set_joint_position(p)

    def set_joint_target_position(self, value: float):
        for j, p in zip(self.gripper_joints, self.__to_joint_positions(value)):
            j.set_joint_target_position(p)

    def get_joint_position(self) -> float:
        return self.__to_gripper_position(np.array([j.get_joint_position() for j in self.gripper_joints]))

    def get_joint_target_position(self) -> float:
        return self.__to_gripper_position(np.array([j.get_joint_target_position() for j in self.gripper_joints]))

    def get_joint_velocity(self) -> float:
        limits_lower, limits_upper = self.__joint_limits()
        return np.mean([j.get_joint_velocity() / (limits_upper - limits_lower) for j in self.gripper_joints]).item()

    def get_joint_target_velocity(self) -> float:
        limits_lower, limits_upper = self.__joint_limits()
        return np.mean(
            j.get_joint_target_velocity() / (limits_upper - limits_lower) for j in self.gripper_joints).item()

    def set_joint_target_velocity(self, value: float):
        limits_lower, limits_upper = self.__joint_limits()
        for j, p in zip(self.gripper_joints, value * (limits_upper - limits_lower)):
            j.set_joint_target_velocity(p)

    def get_torque(self) -> float:
        return np.mean(j.get_torque() for j in self.gripper_joints).item()

    def set_torque(self, value: float):
        for j in self.gripper_joints:
            j.set_torque(value)

    def set_mode(self, mode: JointMode):
        for j in self.gripper_joints:
            j.set_mode(mode)

    def get_mode(self) -> JointMode:
        if len(self.gripper_joints) == 0:
            return JointMode.POSITION_CONTROL
        return self.gripper_joints[0].get_mode()

    def get_upper_velocity_limit(self) -> float:
        return max([j.get_upper_velocity_limit() for j in self.gripper_joints])

    def get_joint_interval(self) -> Tuple[float, float]:
        return 0, 1
