from .reward import Reward


class ConstantReward(Reward):
    """
    A constant reward.
    """

    def __init__(self, value: float):
        """
        :param value:  Value to return at each time step.
        """
        super().__init__("constant_reward", name_abbreviation="const")
        self.__value = value

    def calculate(self) -> float:
        return self.__value

    def __repr__(self):
        return "{:0.2e}".format(self.__value)

    @property
    def value(self) -> float:
        return self.__value
