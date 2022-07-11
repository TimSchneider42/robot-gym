from typing import Tuple, TYPE_CHECKING

import numpy as np

from .normalized_reward import NormalizedReward
if TYPE_CHECKING:
    from robot_gym.core import BaseTask


class ActionReward(NormalizedReward["BaseTask"]):
    def __init__(self):
        super(ActionReward, self).__init__(name="action_reward")

    def _calculate_reward_unnormalized(self) -> float:
        action = self.task.latest_action
        action_arr = np.concatenate(list(action.values()))
        return -(action_arr ** 2).sum()

    def _get_reward_bounds(self) -> Tuple[float, float]:
        action_names = list(self.task.action_space.spaces.keys())
        action_lims_upper = np.concatenate([self.task.action_space[n].high for n in action_names])
        action_lims_lower = np.concatenate([self.task.action_space[n].low for n in action_names])
        min_squared_value = np.minimum(action_lims_upper ** 2, action_lims_lower ** 2)
        zero_in_interval = np.logical_and(action_lims_lower <= 0, action_lims_upper >= 0)
        min_squared_value[zero_in_interval] = 0
        max_squared_value = np.maximum(action_lims_upper ** 2, action_lims_lower ** 2)
        return -max_squared_value.sum(), -min_squared_value.sum()
