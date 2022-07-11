from typing import TypeVar, Generic, Optional, Callable

from .reward import Reward
from robot_gym.core import BaseTask

TaskType = TypeVar("TaskType", bound=BaseTask)


class LambdaReward(Reward[TaskType], Generic[TaskType]):
    """
    A reward class that uses a lambda function to calculate the reward.
    """

    def __init__(self, reward_fun: Callable[[TaskType], float], name: str = "lambda",
                 name_abbreviation: Optional[str] = None):
        """
        :param reward_fun:                          a function that takes the gym environment as input and produces a
                                                    reward for the current time step
        :param name:                                the name of the reward (to be used as key in info dictionary
                                                    returned by the gym environment every step)
        :param name_abbreviation:                   an abbreviation of the name of the reward that is displayed with the
                                                    value of the reward at the end of each episode (by the gym
                                                    environment)
        """
        super().__init__(name, name_abbreviation=name_abbreviation)
        self.__reward_fun = reward_fun

    def calculate(self) -> float:
        return self.__reward_fun(self.task)
