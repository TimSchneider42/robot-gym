from typing import Dict, Tuple

import numpy as np

from .continuous_sensor import ContinuousSensor
from robot_gym.core import BaseTask


class PreviousActionSensor(ContinuousSensor[BaseTask]):
    def __init__(self):
        super(PreviousActionSensor, self).__init__()

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {"prev_act_" + k: (v.low, v.high) for k, v in self.task.action_space.spaces.items()}

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return self._observe_unnormalized()

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        if self.task.latest_action is None:
            return {"prev_act_" + k: np.zeros(v.shape) for k, v in self.task.action_space.spaces.items()}
        else:
            return {"prev_act_" + k: v for k, v in self.task.latest_action.items()}
