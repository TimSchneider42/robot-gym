from typing import Tuple, TYPE_CHECKING

import numpy as np

from .normalized_reward import NormalizedReward

if TYPE_CHECKING:
    from robot_gym.core.reacher_task import ReacherTask


class ReacherReward(NormalizedReward["ReacherTask"]):
    """
    A reward for stabilizing the end-effector at a given target position. To be used only with ReacherTask.
    """

    def __init__(self, robot_name: str = "ur10", max_pos_distance: float = 1.0):
        """
        :param max_pos_distance:                    the maximum distance to use for normalizing the (unscaled) reward to
                                                    lie in [-1, 0]
        """
        super().__init__(name="reacher_reward", clip=False)
        self.__max_distance = self.__ssd_distance(max_pos_distance ** 2)
        self.__min_distance = self.__ssd_distance(0)
        self.__robot_name = robot_name

    def _calculate_reward_unnormalized(self) -> float:
        euclidean_distance = np.asscalar(np.sum(
            (self.task.environment.robots[self.__robot_name].gripper.pose.translation
             - self.task.target_position_world_frame) ** 2))
        return -self.__ssd_distance(euclidean_distance)

    @staticmethod
    def __ssd_distance(euclidean_distance: float) -> float:
        return euclidean_distance + 1e-2 * (np.log(euclidean_distance + 1e-5) - np.log(1e-5))

    def _get_reward_bounds(self) -> Tuple[float, float]:
        pass
