from typing import Generic, TypeVar, Optional, Tuple, Iterable, TYPE_CHECKING

from .reward import Reward

if TYPE_CHECKING:
    from robot_gym.core import BaseTask

TaskType = TypeVar("TaskType", bound="BaseTask")


class RewardSum(Reward[TaskType], Generic[TaskType]):
    """
    A sum of rewards.
    """

    def __init__(self, summands: Iterable[Reward[TaskType]], name: Optional[str] = "sum",
                 name_abbreviation: Optional[str] = None):
        """
        :param summands: Summands to add.
        """
        self.__summands = tuple(summands)

        super().__init__(name, name_abbreviation=name_abbreviation)

    def _on_initialize(self):
        for s in self.__summands:
            s.initialize(self.task)

    def calculate(self) -> float:
        return sum([s() for s in self.__summands])

    def reset(self):
        for s in self.__summands:
            s.reset()

    def __repr__(self):
        return " + ".join(map(str, self.__summands))

    @property
    def summands(self) -> Tuple[Reward, ...]:
        return self.__summands
