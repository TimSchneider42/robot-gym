from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Generic, Dict

from .object import Object
from .robot import Robot


def ensure_initialized(func):
    def wrapped(self, *args, **kwargs):
        assert self.initialized, "{} can only be called after the environment is initialized".format(func.__name__)
        return func(self, *args, **kwargs)

    return wrapped


ObjectType = TypeVar("ObjectType", bound=Object)
RobotType = TypeVar("RobotType", bound=Robot)


class Environment(ABC, Generic[ObjectType, RobotType]):
    def __init__(self, display_name: str):
        self.__initialized = False
        self.__time_step = None
        self.__display_name = display_name
        self.__robots: Optional[Dict[str, RobotType]] = None
        self.__synchronized_mode = True

    def initialize(self, time_step: float):
        self.__robots = {}
        self.__time_step = time_step
        self._initialize(time_step)
        self.__initialized = True

    def _register_robot(self, robot: RobotType):
        name = robot.name
        assert name not in self.__robots, "Robot {} already exists!".format(name)
        self.__robots[name] = robot

    def _unregister_robot(self, name: str):
        del self.__robots[name]

    def _initialize(self, time_step: float):
        pass

    @ensure_initialized
    def terminate_episode(self):
        """
        Clean-up at the end of each episode, e.g. for stopping the robot (in the real environment).
        """
        pass

    @ensure_initialized
    def start_reset(self):
        pass

    @ensure_initialized
    def terminate_reset(self):
        pass

    @ensure_initialized
    def start_episode(self):
        pass

    @ensure_initialized
    def shutdown(self):
        pass

    @ensure_initialized
    def step(self):
        self._step()

    def _step(self):
        pass

    @property
    def initialized(self):
        return self.__initialized

    @property
    def robots(self) -> Optional[Dict[str, RobotType]]:
        return self.__robots

    @property
    def display_name(self) -> str:
        return self.__display_name

    @property
    def time_step(self) -> float:
        return self.__time_step

    @property
    def synchronized_mode(self):
        """
        Only for the real system. Do not wait for the next control cycle, but rather execute actions straight away when
        Environment.step() is called.
        """
        return self.__synchronized_mode

    @synchronized_mode.setter
    def synchronized_mode(self, value: bool):
        self.__synchronized_mode = value

    def __repr__(self) -> str:
        return self.__display_name
