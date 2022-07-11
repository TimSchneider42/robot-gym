from abc import ABC, abstractmethod

from robot_gym.environment.generic import Object
from transformation import Transformation


class SimulationObject(Object, ABC):
    @abstractmethod
    def set_pose(self, value: Transformation):
        pass

    @abstractmethod
    def set_collidable(self, value: bool):
        pass

    @abstractmethod
    def set_static(self, value: bool):
        pass
