from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .robot_component import RobotComponent
from transformation import Transformation


class RobotArm(RobotComponent, ABC):
    @abstractmethod
    def move_to_pose(self, target_pose: Transformation):
        """
        Moves to the target pose, blocking until it is reached.
        :param target_pose:
        :return:
        """
        pass

    @abstractmethod
    def move_towards_pose_linear(self, target_pose: Transformation, linear_velocity: float = 0.25,
                                 angular_velocity: Optional[float] = None):
        """
        Starts moving towards the target pose (linear in cartesian coordinates) in a synchronized matter when
        environment.step() is called.
        :param target_pose:         Target pose to reach.
        :param linear_velocity:     Linear target velocity.
        :param angular_velocity:    Angular target velocity. If this value is set, the target rotation might be reached
                                    at a different time than the target translation.
        :return:
        """
        pass

    @abstractmethod
    def move_cartesian_velocity(self, linear_vel: np.ndarray, angular_vel: np.ndarray):
        pass

    @abstractmethod
    def stop_linear_control(self):
        pass
