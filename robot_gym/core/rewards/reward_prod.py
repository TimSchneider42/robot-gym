import operator
from functools import reduce
from typing import Generic, TypeVar, Optional, Tuple, Iterable, TYPE_CHECKING

from .reward import Reward

if TYPE_CHECKING:
    from robot_gym.core import BaseTask

TaskType = TypeVar("TaskType", bound="BaseTask")


class RewardProd(Reward[TaskType], Generic[TaskType]):
    """
    A sum of rewards.
    """

    def __init__(self, factors: Iterable[Reward], name: Optional[str] = "prod",
                 name_abbreviation: Optional[str] = None):
        """
        :param factors: Summands to add.
        """
        self.__factors = tuple(factors)

        super().__init__(name, name_abbreviation=name_abbreviation)

    def _on_initialize(self):
        for f in self.__factors:
            f.initialize(self.task)

    def calculate(self) -> float:
        return reduce(operator.mul, [f() for f in self.__factors])

    def reset(self):
        for f in self.__factors:
            f.reset()

    def __repr__(self):
        return "(" + ") * (".join(map(str, self.__factors)) + ")"

    @property
    def factors(self) -> Tuple[Reward, ...]:
        return self.__factors
