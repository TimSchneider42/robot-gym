from abc import ABC
from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

from .joint_sensor import JointSensor
from robot_gym.core.sensors import ContinuousSensor

if TYPE_CHECKING:
    from robot_gym.core import BaseTask


class JointSinCosSensor(ContinuousSensor["BaseTask"], JointSensor, ABC):
    def __init__(self, *args, **kwargs):
        super(JointSinCosSensor, self).__init__(*args, **kwargs)

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        num = len(self.observed_robot_component.joint_intervals)
        return {
            "{}_joint_pos_sin".format(self.name_prefix): (-np.ones(num), np.ones(num)),
            "{}_joint_pos_cos".format(self.name_prefix): (-np.ones(num), np.ones(num))
        }

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        pos = self.observed_robot_component.joint_positions
        return {
            "{}_joint_pos_sin".format(self.name_prefix): np.sin(pos),
            "{}_joint_pos_cos".format(self.name_prefix): np.cos(pos)
        }

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return self._observe_unnormalized()
