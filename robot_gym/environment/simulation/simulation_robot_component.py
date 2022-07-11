from abc import ABC, abstractmethod
from typing import Tuple, TypeVar

import numpy as np

from .simulation_joint import SimulationJoint
from .simulation_object import SimulationObject
from robot_gym.environment.generic import JointMode, RobotComponent

JointType = TypeVar("JointType", bound=SimulationJoint)


class SimulationRobotComponent(SimulationObject, RobotComponent, ABC):
    @abstractmethod
    def get_joints(self) -> Tuple[JointType]:
        pass

    def set_joint_target_velocities(self, velocities: np.ndarray):
        for j, v in zip(self.get_joints(), velocities):
            j.set_joint_target_velocity(v)

    def set_joint_target_positions(self, positions: np.ndarray):
        for j, p in zip(self.get_joints(), positions):
            j.set_joint_target_position(p)

    def move_to_joint_positions(self, positions: np.ndarray):
        for j, p in zip(self.get_joints(), positions):
            j.set_joint_position(p)

    def set_joint_torques(self, torques: np.ndarray):
        for j, t in zip(self.get_joints(), torques):
            j.set_torque(t)

    def set_joint_mode(self, mode: JointMode):
        for j in self.get_joints():
            j.set_mode(mode)

    def get_joint_positions(self) -> np.ndarray:
        return np.array([j.get_joint_position() for j in self.get_joints()])

    def get_joint_velocities(self) -> np.ndarray:
        return np.array([j.get_joint_velocity() for j in self.get_joints()])

    def get_joint_target_positions(self) -> np.ndarray:
        return np.array([j.get_joint_target_position() for j in self.get_joints()])

    def get_joint_target_velocities(self) -> np.ndarray:
        return np.array([j.get_joint_target_velocity() for j in self.get_joints()])

    def get_joint_velocity_limits(self):
        return np.array([j.get_upper_velocity_limit() for j in self.get_joints()])

    def get_joint_intervals(self) -> np.ndarray:
        return np.array([j.get_joint_interval() for j in self.get_joints()])

    def get_joint_mode(self) -> JointMode:
        return self.get_joints()[0].get_mode()

    @property
    def joints(self) -> Tuple[JointType]:
        return self.get_joints()
