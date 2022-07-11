from typing import Tuple, Optional

from .pybullet_object import PybulletObject
from pyboolet.multibody import Multibody
from pyboolet.simulation_object import Link
from transformation import Transformation
from robot_gym.environment.simulation import SimulationRobotComponent, SimulationJoint


class PybulletRobotComponent(SimulationRobotComponent, PybulletObject):
    def __init__(self, wrapped_body: Multibody, joints: Tuple[SimulationJoint, ...],
                 wrapped_root_link: Optional[Link] = None,
                 virtual_root_link_transformation: Optional[Transformation] = None,
                 boundary_link: Optional[Link] = None):
        PybulletObject.__init__(self, wrapped_body, wrapped_root_link=wrapped_root_link,
                                virtual_root_link_transformation=virtual_root_link_transformation,
                                boundary_link=boundary_link)
        self.__joints = joints

    def get_joints(self) -> Tuple[SimulationJoint, ...]:
        return self.__joints

