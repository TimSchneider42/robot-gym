from abc import ABC, abstractmethod
from typing import Tuple

from robot_gym.environment.generic.joint_mode import JointMode


class SimulationJoint(ABC):
    @abstractmethod
    def get_joint_position(self) -> float:
        pass

    @abstractmethod
    def set_joint_position(self, value: float):
        pass

    @abstractmethod
    def get_joint_velocity(self) -> float:
        pass

    @abstractmethod
    def set_joint_target_position(self, value: float):
        pass

    @abstractmethod
    def get_joint_target_velocity(self) -> float:
        pass

    @abstractmethod
    def set_joint_target_velocity(self, value: float):
        pass

    @abstractmethod
    def get_torque(self) -> float:
        pass

    @abstractmethod
    def set_torque(self, value: float):
        pass

    @abstractmethod
    def set_mode(self, mode: JointMode):
        pass

    @abstractmethod
    def get_mode(self) -> JointMode:
        pass

    @abstractmethod
    def get_joint_target_position(self) -> float:
        pass

    @abstractmethod
    def get_joint_target_velocity(self) -> float:
        pass

    @abstractmethod
    def get_upper_velocity_limit(self) -> float:
        pass

    @abstractmethod
    def get_joint_interval(self) -> Tuple[float, float]:
        pass
