from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from transformation import Transformation


class Object(ABC):
    @abstractmethod
    def get_pose(self) -> Transformation:
        pass

    @abstractmethod
    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @property
    def pose(self) -> Transformation:
        return self.get_pose()

    @property
    def velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_velocity()
