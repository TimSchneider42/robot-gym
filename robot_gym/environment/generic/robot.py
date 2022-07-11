from abc import ABC
from typing import TypeVar, Generic, Tuple

import numpy as np

from .object import Object
from .robot_arm import RobotArm
from .robot_component import RobotComponent
from transformation import Transformation

RobotComponentType = TypeVar("RobotComponentType", bound=RobotComponent)
RobotArmType = TypeVar("RobotArmType", bound=RobotArm)


class Robot(Object, Generic[RobotArmType, RobotComponentType], ABC):
    def __init__(self, arm: RobotArmType, gripper: RobotComponentType, name: str):
        self._arm = arm
        self._gripper = gripper
        self._name = name

    @property
    def arm(self) -> RobotArmType:
        return self._arm

    @property
    def gripper(self) -> RobotComponentType:
        return self._gripper

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return "Robot {}".format(self._name)

    def get_pose(self) -> Transformation:
        return self._arm.get_pose()

    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._arm.get_velocity()

