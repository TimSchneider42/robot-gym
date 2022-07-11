import itertools
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Dict, Iterable, TYPE_CHECKING, Optional, TypeVar, Generic, Any

import gym
import numpy as np

from robot_gym.environment.generic import Environment
from robot_gym import logger

if TYPE_CHECKING:
    from .controllers import Controller
    from .sensors import Sensor
    from .rewards import Reward, RewardSum


class EpisodeInvalidException(Exception):
    """
    Exception that is raised when an error occurs that invalidates the measurements of the current episode (e.g. when
    the tracking of an object is temporarily lost). The task is expected to recover from this error once reset() is
    called.
    """
    pass


class WrappedEpisodeInvalidException(EpisodeInvalidException):
    def __init__(self, inner_exception: Optional[Exception] = None):
        self.__inner_exception = inner_exception
        super(EpisodeInvalidException, self).__init__()

    @property
    def inner_exception(self):
        return self.__inner_exception

    def __repr__(self):
        return "The current episode is invalid because the following error occurred: {}".format(self.__inner_exception)


EnvironmentType = TypeVar("EnvironmentType", bound=Environment)


class TaskState(Enum):
    AWAITING_INITIALIZE = 0
    AWAITING_RESET = 1
    RESET_DONE = 2
    DURING_EPISODE = 3


class BaseTask(gym.Env, ABC, Generic[EnvironmentType]):
    """
    An abstract base class for gym environments that use the robot (real or simulated).
    """

    def __init__(self, controllers: Iterable["Controller[BaseTask]"],
                 sensors: Iterable["Sensor[BaseTask]"], reward: "Reward[BaseTask]",
                 time_step: float, time_limit_steps: Optional[int] = None):
        """
        :param controllers:             A sequence of controller objects that define the actions on the environment
        :param sensors:                 A sequence of sensors objects that provide observations to the agent
        :param reward:                  Reward for this task.
        :param time_step:               The time between two controller updates (actions of the agent)
        :param time_limit_steps:        The number of steps until the episode terminates (if no other termination
                                        criterion is reached)
        """
        self.__cumulative_rewards_dict: Dict["Reward", float] = {}

        self.__sensors: Iterable[Sensor] = sensors
        self.__reward = reward
        from robot_gym.core.rewards import RewardSum
        if isinstance(self.__reward, RewardSum):
            self.__wrapped_reward = reward
        else:
            self.__wrapped_reward = RewardSum([reward])

        self.__time_limit_steps: int = time_limit_steps
        self.__current_step: Optional[int] = None  # needed to enforce time_limit_steps

        self.__env: Optional[EnvironmentType] = None

        self.__time_step: float = time_step

        self.__controllers = tuple(controllers)

        self.__latest_action: Optional[Dict[str, np.ndarray]] = None
        self.__state = TaskState.AWAITING_INITIALIZE

        self.__done = False

    def initialize(self, env: EnvironmentType):
        """
        Initialize the gym environment. Needs to be called before reset() is called for the first time.

        :param env:             the environment object that should be used to interact with the (simulated or real)
                                scene
        """
        self.__assert_state_is(TaskState.AWAITING_INITIALIZE)
        self.__env = env
        self.__env.initialize(self.__time_step)

        self._initialize()

        action_space_dict = {}
        for controller in self.__controllers:
            new_action_space = controller.initialize(self)
            assert controller.name not in action_space_dict, "Duplicate action name"
            action_space_dict[controller.name] = new_action_space
        self.action_space = gym.spaces.Dict(action_space_dict)

        obs_space_dict = {}
        for sensor in self.__sensors:
            new_obs_spaces = sensor.initialize(self)
            assert len(set(new_obs_spaces.keys()).intersection(set(obs_space_dict.keys()))) == 0, \
                "Duplicate observation name"
            obs_space_dict.update(new_obs_spaces)
        self.observation_space = gym.spaces.Dict(obs_space_dict)

        self.__wrapped_reward.initialize(self)

        self.__done = False
        self.__state = TaskState.AWAITING_RESET

    def _initialize(self):
        pass

    def __assert_state_is(self, expected_state: TaskState):
        assert self.__state == expected_state, "Expected task state to be {}, got {}.".format(
            expected_state.name, self.__state.name)

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Executes one action of the agent and calculates the resulting observation, reward, and done signal.

        :param action:          the action that should be executed
        :return:                the feedback from the environment as a (observation, reward, done, info) tuple
        """
        if self.__state == TaskState.RESET_DONE:
            self.environment.start_episode()
            self.__state = TaskState.DURING_EPISODE
        self.__assert_state_is(TaskState.DURING_EPISODE)
        self.__current_step += 1
        self.__latest_action = action

        for controller in self.__controllers:
            controller.actuate(action[controller.name])

        self.__env.step()

        self.__done, task_info = self._step_task()
        if self.__time_limit_steps is not None:
            self.__done |= self.__time_limit_steps <= self.__current_step

        current_rewards = {reward: reward() for reward in self.__wrapped_reward.summands}
        reward = sum(current_rewards.values())

        reward_info = {reward.name: value for reward, value in current_rewards.items()}
        info = {
            "reward": reward_info,
            **task_info
        }

        obs = {k: v for k, v in itertools.chain(*(sensor.observe().items() for sensor in self.__sensors))}

        for reward_obj, value in current_rewards.items():
            self.__cumulative_rewards_dict[reward_obj] += value

        if self.__done:
            self.terminate_episode()

            cumulative_reward = sum(self.__cumulative_rewards_dict.values())
            current_reward = sum(current_rewards.values())
            logger.info("Cumulative reward: {: .6f} ({})".format(
                cumulative_reward,
                "  ".join([
                    "{}: {: .6f} [{: 8.4f}%]".format(r.name_abbreviation, v, v / max(cumulative_reward, 1e-4) * 100)
                    for r, v in self.__cumulative_rewards_dict.items()])))
            logger.info("Final reward:      {: .6f} ({})".format(
                current_reward,
                "  ".join([
                    "{}: {: .6f} [{: 8.4f}%]".format(r.name_abbreviation, v, v / max(current_reward, 1e-4) * 100)
                    for r, v in current_rewards.items()])))
            for h in logger.handlers:
                h.flush()
        return obs, reward, self.__done, info

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Resets the gym environment. Must be called at the beginning of each episode.

        :return:        the initial observation for the episode
        """
        if self.__state in [TaskState.DURING_EPISODE, TaskState.RESET_DONE]:
            self.terminate_episode()
        self.__assert_state_is(TaskState.AWAITING_RESET)
        self.__current_step = 0
        self.__latest_action = None
        self.__env.start_reset()

        for controller in self.__controllers:
            controller.initialize(self)

        self._reset_task()
        self.__wrapped_reward.reset()

        self.__cumulative_rewards_dict = {reward: 0 for reward in self.__wrapped_reward.summands}

        return_val = {
            k: v for k, v in itertools.chain(*(sensor.reset().items() for sensor in self.__sensors))}

        self.__done = False
        self.environment.terminate_reset()
        self.__state = TaskState.RESET_DONE
        return return_val

    def close(self):
        """
        Shuts the environment down. Should be called if the environment is not used anymore.
        """
        self.__env.shutdown()

    def _restart_env(self):
        """
        Restarts the environment in case an error occurred.
        """
        pass

    @abstractmethod
    def _step_task(self) -> Tuple[bool, Dict]:
        """
        Execute the task-specific components of gym.step().

        :return:                        a (done, info) tuple, where info contains the task-specific
                                        components of the infos, respectively
        """
        pass

    @abstractmethod
    def _reset_task(self):
        """
        Execute the task-specific components of gym.reset().
        """
        pass

    def terminate_episode(self):
        self.__state = TaskState.AWAITING_RESET
        self.environment.terminate_episode()

    @property
    def controllers(self) -> Tuple["Controller", ...]:
        return self.__controllers

    @property
    def sensors(self) -> Iterable["Sensor"]:
        return self.__sensors

    @property
    def reward(self) -> "Reward":
        return self.__reward

    @property
    def environment(self) -> EnvironmentType:
        return self.__env

    @property
    def time_step(self) -> float:
        return self.__time_step

    @property
    def time_limit_steps(self) -> int:
        return self.__time_limit_steps

    @property
    def current_step(self) -> int:
        return self.__current_step

    @property
    def latest_action(self) -> Optional[Dict[str, np.ndarray]]:
        return self.__latest_action

    @property
    def done(self) -> bool:
        return self.__done
