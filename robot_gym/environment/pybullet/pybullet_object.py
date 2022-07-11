from typing import Tuple, Optional

import numpy as np

from pyboolet.multibody import Multibody
from pyboolet.simulation_object import Link
from robot_gym.environment.simulation import SimulationObject
from transformation import Transformation


class PybulletObject(SimulationObject):
    def __init__(self, wrapped_body: Multibody,
                 wrapped_root_link: Optional[Link] = None,
                 virtual_root_link_transformation: Optional[Transformation] = None,
                 boundary_link: Optional[Link] = None):
        self.__wrapped_root_link = wrapped_root_link
        self.__wrapped_body = wrapped_body
        self.__virtual_root_link_transformation = virtual_root_link_transformation
        self.__boundary_link = boundary_link

    def get_pose(self) -> Transformation:
        if self.__wrapped_root_link is None:
            pose = self.__wrapped_body.pose
        else:
            pose = self.__wrapped_root_link.pose
        if self.__virtual_root_link_transformation is not None:
            pose = pose.transform(self.__virtual_root_link_transformation)
        return pose

    def set_pose(self, value: Transformation):
        assert self.__wrapped_root_link is None, "Can only set pose of a object at the root of a kinematic tree"
        self.__wrapped_body.reset_pose(value)
        self.__wrapped_body.reset_velocity(np.zeros(3), np.zeros(3))

    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.__wrapped_root_link is None:
            vel = self.__wrapped_body.velocity
        else:
            vel = self.__wrapped_root_link.velocity
        if self.__virtual_root_link_transformation is not None:
            v_trans = self.__virtual_root_link_transformation.translation
            if np.linalg.norm(v_trans) > 0:
                # Compute additional linear velocity induced by angular velocity
                vel_lin, vel_ang = vel
                additional_vel_dir = np.cross(vel_ang, v_trans)
                additional_vel_dir /= np.linalg.norm(additional_vel_dir)
                additional_vel = additional_vel_dir * np.linalg.norm(vel_ang) * np.linalg.norm(v_trans)
                vel = vel_lin + additional_vel, vel_ang
        return vel

    def set_collidable(self, value: bool):
        if self.__wrapped_root_link is None:
            self.__wrapped_body.set_base_collidable(value)
            if self.__wrapped_body.root_link is not None:
                self.__set_collidable_recursive(self.__wrapped_body.root_link, value)
        else:
            self.__set_collidable_recursive(self.__wrapped_root_link, value)

    def set_static(self, value: bool):
        if self.__wrapped_root_link is None:
            self.__wrapped_body.base_static = value
            if self.__wrapped_body.root_link is not None:
                self.__set_static_recursive(self.__wrapped_body.root_link, value)
        else:
            self.__set_static_recursive(self.__wrapped_root_link, value)

    def __set_static_recursive(self, root: Link, value: bool):
        for c in root.child_joints:
            c.soft_fixed = value
            if c.child is not self.__boundary_link:
                self.__set_static_recursive(c.child, value)

    def __set_collidable_recursive(self, root: Link, value: bool):
        root.set_collidable(value)
        for c in root.child_joints:
            if c.child is not self.__boundary_link:
                self.__set_collidable_recursive(c.child, value)

    @property
    def wrapped_root_link(self) -> Optional[Link]:
        return self.__wrapped_root_link

    @property
    def boundary_link(self) -> Optional[Link]:
        return self.__boundary_link

    @property
    def wrapped_body(self) -> Multibody:
        return self.__wrapped_body

    def _update_wrapped_body(self, new_wrapped_body: Multibody, new_wrapped_link: Optional[Link] = None):
        self.__wrapped_root_link = new_wrapped_link
        self.__wrapped_body = new_wrapped_body
