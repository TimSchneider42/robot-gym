from typing import Sequence

from .pybullet_joint import PybulletJoint
from robot_gym.environment.simulation.simulation_gripper_joint import SimulationGripperJoint


class PybulletGripperJoint(SimulationGripperJoint[PybulletJoint]):
    def __init__(self, gripper_joints: Sequence[PybulletJoint]):
        self.__gripper_joints = tuple(gripper_joints)

    def get_gripper_joints(self) -> Sequence[PybulletJoint]:
        return self.__gripper_joints
