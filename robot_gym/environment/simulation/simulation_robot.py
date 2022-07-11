from abc import ABC
from typing import TypeVar

from .simulation_object import SimulationObject
from .simulation_robot_component import SimulationRobotComponent
from robot_gym.environment.generic import Robot
from robot_gym.environment.generic.robot_arm import RobotArm

RobotComponentType = TypeVar("RobotComponentType", bound=SimulationRobotComponent)
RobotArmType = TypeVar("RobotArmType", bound=RobotArm)


class SimulationRobot(SimulationObject, Robot[RobotArmType, RobotComponentType], ABC):
    def __init__(self, arm: RobotArmType, gripper: RobotComponentType, name: str):
        SimulationObject.__init__(self)
        Robot.__init__(self, arm, gripper, name)
