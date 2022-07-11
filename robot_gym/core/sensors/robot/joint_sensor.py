from abc import abstractmethod, ABC
from typing import Optional, TYPE_CHECKING

from robot_gym.core.sensors import Sensor
from robot_gym.environment.generic import RobotComponent

if TYPE_CHECKING:
    from robot_gym.core import BaseTask


class JointSensor(Sensor["BaseTask"], ABC):
    def __init__(self, name_prefix: str, **kwargs):
        super(JointSensor, self).__init__(**kwargs)
        self.__name_prefix = name_prefix
        self.__observed_robot_component: Optional[RobotComponent] = None

    @abstractmethod
    def _get_observed_robot_component(self) -> RobotComponent:
        pass

    @property
    def observed_robot_component(self) -> RobotComponent:
        return self._get_observed_robot_component()

    @property
    def name_prefix(self) -> str:
        return self.__name_prefix
