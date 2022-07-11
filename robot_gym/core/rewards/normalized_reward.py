from abc import abstractmethod
from typing import TypeVar, Generic, Optional, Tuple

from .reward import Reward
from robot_gym.core import BaseTask

TaskType = TypeVar("TaskType", bound=BaseTask)


class NormalizedReward(Reward[TaskType], Generic[TaskType]):
    """
    An abstract base class for rewards. Implementing classes should make sure that the reward lies in [-1, 0], so that
    all rewards have the same scaling and weight (scaling) factors can be chosen easier. These scaling factors are used
    to weigh different rewards relative to each other.
    """

    def __init__(self, clip: bool = False, name: str = "normalized_reward", name_abbreviation: Optional[str] = None):
        """
        :param clip:                                whether the (unnormalized, unscaled) reward should be clipped to not
                                                    go lower than the value returned by _get_min_reward_unnormalized()
        :param name:                                the name of the reward (to be used as key in info dictionary
                                                    returned by the gym environment every step)
        :param name_abbreviation:                   an abbreviation of the name of the reward that is displayed with the
                                                    value of the reward at the end of each episode (by the gym
                                                    environment)
        """
        super(NormalizedReward, self).__init__(name, name_abbreviation=name_abbreviation)
        self.__clip: bool = clip
        self.__min_reward: Optional[float] = None
        self.__max_reward: Optional[float] = None

    def _on_initialize(self):
        self.__min_reward, self.__max_reward = self._get_reward_bounds()

    def calculate(self) -> float:
        """
        Calculates the reward for the current time step.

        :return:            the reward for the current time step
        """
        reward = self._calculate_reward_unnormalized()
        if self.__clip:
            reward = min(max(reward, self.__min_reward), self.__max_reward)
        return (reward - self.__min_reward) / (self.__max_reward - self.__min_reward)

    @abstractmethod
    def _calculate_reward_unnormalized(self) -> float:
        """
        Calculates the unnormalized reward for the current time step.
        """

    @abstractmethod
    def _get_reward_bounds(self) -> Tuple[float, float]:
        """
        Returns a tuple consisting of the lower limit and the upper limit of the value this reward can take.
        These values are used to normalize the reward.
        :return: a tuple (lower_lim, upper_lim)
        """
