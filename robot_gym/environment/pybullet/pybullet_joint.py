from typing import Tuple

from robot_gym.environment.simulation import SimulationJoint
from robot_gym.environment.generic import JointMode
from pyboolet.simulation_object import JointControlMode, RevoluteJoint


class PybulletJoint(SimulationJoint):
    def __init__(self, wrapped_joint: RevoluteJoint):
        self.__wrapped_joint = wrapped_joint

    def get_joint_position(self) -> float:
        return self.__wrapped_joint.joint_position

    def set_joint_position(self, value: float):
        self.__wrapped_joint.reset_joint_state(position=value, velocity=0.0)

    def get_joint_velocity(self) -> float:
        return self.__wrapped_joint.joint_velocity

    def get_joint_target_position(self) -> float:
        return self.__wrapped_joint.target_position

    def set_joint_target_position(self, value: float):
        self.__wrapped_joint.target_position = value

    def get_joint_target_velocity(self) -> float:
        return self.__wrapped_joint.target_velocity

    def set_joint_target_velocity(self, value: float):
        self.__wrapped_joint.target_velocity = value

    def get_torque(self) -> float:
        return self.__wrapped_joint.torque

    def set_torque(self, value: float):
        self.__wrapped_joint.torque = value

    def set_mode(self, mode: JointMode):
        self.__wrapped_joint.control_mode = JointControlMode(mode.value)

    def get_mode(self) -> JointMode:
        return JointMode(self.__wrapped_joint.control_mode.value)

    def get_upper_velocity_limit(self) -> float:
        return self.__wrapped_joint.max_velocity

    def get_joint_interval(self) -> Tuple[float, float]:
        return self.__wrapped_joint.interval

    @property
    def wrapped_joint(self) -> RevoluteJoint:
        return self.__wrapped_joint
