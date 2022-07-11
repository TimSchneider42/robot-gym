from abc import ABC, abstractmethod

import numpy as np

from .joint_mode import JointMode
from .object import Object


class RobotComponent(Object, ABC):
    """
    Units
        - cartesian properties: m, m/s
        - joint properties:
            - revolute: rad, rad/s
            - prismatic: m, m/s
    """

    @abstractmethod
    def get_joint_target_velocities(self) -> np.ndarray:
        pass

    @abstractmethod
    def set_joint_target_velocities(self, velocities: np.ndarray):
        pass

    @abstractmethod
    def get_joint_target_positions(self) -> np.ndarray:
        pass

    @abstractmethod
    def set_joint_target_positions(self, positions: np.ndarray):
        pass

    @abstractmethod
    def set_joint_torques(self, torques: np.ndarray):
        pass

    @abstractmethod
    def set_joint_mode(self, mode: JointMode):
        pass

    @abstractmethod
    def get_joint_mode(self) -> JointMode:
        pass

    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_joint_velocities(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_joint_velocity_limits(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_joint_intervals(self) -> np.ndarray:
        """
        Returns the joint intervals (i.e. minimum and maximum joint position) for all joints of the robot component.

        :return:            an array joint_intervals, where joint_intervals[i][0] is the minimum joint position and
                            joint_intervals[i][1] the maximum joint position for joint i
        """
        pass

    @property
    def joint_target_positions(self) -> np.ndarray:
        return self.get_joint_target_positions()

    @property
    def joint_target_velocities(self) -> np.ndarray:
        return self.get_joint_target_velocities()

    @property
    def joint_positions(self) -> np.ndarray:
        return self.get_joint_positions()

    @property
    def joint_velocities(self) -> np.ndarray:
        return self.get_joint_velocities()

    @property
    def joint_velocity_limits(self) -> np.ndarray:
        return self.get_joint_velocity_limits()

    @property
    def joint_intervals(self) -> np.ndarray:
        """
        Returns the joint intervals (i.e. minimum and maximum joint position) for all joints of the robot component.

        :return:            an array joint_intervals, where joint_intervals[i][0] is the minimum joint position and
                            joint_intervals[i][1] the maximum joint position for joint i
        """
        return self.get_joint_intervals()

    @property
    def joint_mode(self) -> JointMode:
        return self.get_joint_mode()

    @joint_mode.setter
    def joint_mode(self, value: JointMode):
        self.set_joint_mode(value)

    @joint_target_positions.setter
    def joint_target_positions(self, value: np.ndarray):
        self.set_joint_target_positions(value)

    @joint_target_velocities.setter
    def joint_target_velocities(self, value: np.ndarray):
        self.set_joint_target_velocities(value)
