from typing import Dict, List, Union, Tuple

import numpy as np

from robot_gym.environment import RobotArm
from robot_gym.environment.generic import JointMode
from robot_gym.core import BaseTask
from .robot_component_controller import RobotComponentController


class EndEffectorVelocityController(RobotComponentController):
    """
    An end-effector velocity-based arm controller. An action consists of a (cartesian) target velocity for the
    end-effector. If a target velocity is not possible (e.g. because of singularities), the closest possible velocity
    (according to the L2 norm) velocity is used).
    """

    def __init__(self, robot_name: str, linear_limits_lower: Union[np.ndarray, float],
                 linear_limits_upper: Union[np.ndarray, float], angular_limits_lower: Union[np.ndarray, float],
                 angular_limits_upper: Union[np.ndarray, float]):
        super().__init__("end_effector_velocity", robot_name, "arm")
        self.__linear_limits_lower: np.ndarray = linear_limits_lower
        self.__linear_limits_upper: np.ndarray = linear_limits_upper
        self.__angular_limits_lower: np.ndarray = angular_limits_lower
        self.__angular_limits_upper: np.ndarray = angular_limits_upper

    def _actuate_denormalized(self, action: np.ndarray):
        rc = self.robot_component
        assert isinstance(rc, RobotArm), "An end-effector velocity controller can only be used for arms"
        rc.move_cartesian_velocity(action[:3], action[3:])

    def _initialize(self, task: BaseTask) -> Tuple[np.ndarray, np.ndarray]:
        self.robot_component.set_joint_mode(JointMode.VELOCITY_CONTROL)
        limits_lower = np.concatenate((self.__linear_limits_lower * np.ones(3),
                                       self.__angular_limits_lower * np.ones(3)))
        limits_upper = np.concatenate((self.__linear_limits_upper * np.ones(3),
                                       self.__angular_limits_upper * np.ones(3)))
        return limits_lower, limits_upper

    @classmethod
    def from_parameters(cls, robot_name: str, parameters: Dict[str, List[float]]) -> "EndEffectorVelocityController":
        """
        Create an EndEffectorVelocityController from an parameters dictionary.

        :param robot_name:                      the name of the robot that is controlled
        :param parameters:                      a dictionary containing the entries end_effector_velocity_limits_lower
                                                and end_effector_velocity_limits_upper
        :param smoothness_penalty_weight:       TODO
        :return:                                an EndEffectorVelocityController with the given parameters
        """
        kwargs = {
            "{}_limits_{}".format(t, l): parameters["end_effector_{}_velocity_limits_{}".format(t, l)]
            for t in ["linear", "angular"]
            for l in ["upper", "lower"]
        }
        return EndEffectorVelocityController(robot_name, **kwargs)
