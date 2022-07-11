import operator
from abc import abstractmethod, ABC
from functools import reduce
from itertools import chain
from typing import TypeVar, Generic, Optional, Union, TYPE_CHECKING

from robot_gym.core import BaseTask

if TYPE_CHECKING:
    from .reward_sum import RewardSum
    from .reward_prod import RewardProd

TaskType = TypeVar("TaskType", bound=BaseTask)


class Reward(ABC, Generic[TaskType]):
    """
    An abstract base class for rewards.
    """

    def __init__(self, name: str = "reward", name_abbreviation: Optional[str] = None):
        """
        :param name:                                the name of the reward (to be used as key in info dictionary
                                                    returned by the gym environment every step)
        :param name_abbreviation:                   an abbreviation of the name of the reward that is displayed with the
                                                    value of the reward at the end of each episode (by the gym
                                                    environment)
        """
        self.__name = name
        self.__name_abbreviation = name if name_abbreviation is None else name_abbreviation
        self.__task: Optional[TaskType] = None

    def initialize(self, task: TaskType):
        """
        Initializes the reward.

        :param task:         the environment in which the reward is used
        """
        self.__task = task
        self._on_initialize()

    def _on_initialize(self):
        pass

    def reset(self):
        """
        Resets the reward. Must be called at the beginning of each episode.
        """
        pass

    @abstractmethod
    def calculate(self) -> float:
        """
        Calculates the reward for the current time step.
        :return: the reward
        """

    @property
    def task(self) -> Optional[TaskType]:
        """
        Returns the gym environment for which the reward is used.

        :return:            the gym environment for which the reward is used
        """
        return self.__task

    @property
    def name(self) -> str:
        """
        Returns the name of the reward.

        :return:            the name of the reward
        """
        return self.__name

    @property
    def name_abbreviation(self) -> str:
        """
        Returns the abbreviation of the name of the reward.

        :return:            the abbreviation of the name of the reward
        """
        return self.__name_abbreviation

    def __repr__(self):
        return "{}".format(type(self).__name__)

    def __call__(self) -> float:
        return self.calculate()

    def __add__(self, other: Union["Reward[TaskType]", float, int]) -> "Reward[TaskType]":
        from .constant_reward import ConstantReward
        from .reward_sum import RewardSum
        if not isinstance(other, Reward):
            other = ConstantReward(float(other))
        s1 = self.summands if isinstance(self, RewardSum) else (self,)
        s2 = other.summands if isinstance(other, RewardSum) else (other,)
        summands = s1 + s2
        non_const = [s for s in summands if not isinstance(s, ConstantReward)]
        const_val = sum(s.value for s in summands if isinstance(s, ConstantReward))
        if len(non_const) == 0:
            return ConstantReward(const_val)
        if const_val != 0:
            simplified_summands = chain(non_const, [ConstantReward(const_val)])
        else:
            simplified_summands = non_const
        return RewardSum(simplified_summands)

    def __radd__(self, other: Union["Reward[TaskType]", float, int]) -> "Reward[TaskType]":
        from .constant_reward import ConstantReward
        if not isinstance(other, Reward):
            other = ConstantReward(float(other))
        return other.__add__(self)

    def __mul__(self, other: Union["Reward[TaskType]", float, int]) -> "Reward[TaskType]":
        from .constant_reward import ConstantReward
        from .reward_prod import RewardProd
        if not isinstance(other, Reward):
            other = ConstantReward(float(other))
        s1 = self.factors if isinstance(self, RewardProd) else (self,)
        s2 = other.factors if isinstance(other, RewardProd) else (other,)
        factors = s1 + s2
        non_const = [f for f in factors if not isinstance(f, ConstantReward)]
        const_val = reduce(operator.mul, (f.value for f in factors if isinstance(f, ConstantReward)))
        if len(non_const) == 0:
            return ConstantReward(const_val)
        if const_val != 0:
            simplified_factors = chain(non_const, [ConstantReward(const_val)])
        else:
            simplified_factors = non_const
        name = abbr = None
        if isinstance(self, ConstantReward):
            if isinstance(other, ConstantReward):
                return ConstantReward(self.value * other.value)
            name = "scaled_" + other.name
            abbr = "s_" + other.name_abbreviation
        return RewardProd(simplified_factors, name=name, name_abbreviation=abbr)

    def __rmul__(self, other: Union["Reward[TaskType]", float, int]) -> "Reward[TaskType]":
        from .constant_reward import ConstantReward
        if not isinstance(other, Reward):
            other = ConstantReward(float(other))
        return other.__mul__(self)
