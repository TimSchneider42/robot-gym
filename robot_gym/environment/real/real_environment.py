import time
from typing import Iterable, Optional, Literal

from robot_gym.environment import Environment, Robot, Object
from robot_gym import logger
from .optitrack import Optitrack
from .rhp12rn_gripper import RHP12RNGripper
from .ur10_robot_arm import UR10RobotArm


class TimingMissedError(Exception):
    def __init__(self, operation: Literal["action", "observation"], missed_by: float):
        self.__operation = operation
        self.__missed_by = missed_by
        super(TimingMissedError, self).__init__("Missed {} timing by {:0.6f}s".format(operation, missed_by))

    @property
    def operation(self):
        return self.__operation

    @property
    def missed_by(self):
        return self.__missed_by


class RealEnvironment(Environment[Object, Robot[UR10RobotArm, RHP12RNGripper]]):
    def __init__(self, optitrack: Optional[Optitrack] = None,
                 robots: Iterable[Robot[UR10RobotArm, RHP12RNGripper]] = (),
                 obs_act_offset_s: float = 0.1, fail_on_timing_missed: bool = False):
        super(RealEnvironment, self).__init__("real-direct")
        self.__optitrack = optitrack
        self.__last_obs_time = None
        self.__obs_act_offset_s = obs_act_offset_s
        self.__robots = tuple(robots)
        self.__fail_on_timing_missed = fail_on_timing_missed

    def _initialize(self, time_step: float):
        super(RealEnvironment, self)._initialize(time_step)
        for r in self.__robots:
            self._register_robot(r)
        if self.__optitrack is not None:
            self.__optitrack.initialize()
        for robot in self.robots.values():
            robot.gripper.initialize()
            robot.arm.initialize(self.__optitrack)
        self._observe()

    def start_reset(self):
        for robot in self.robots.values():
            robot.arm.on_reset_start()
        self._observe()

    def start_episode(self):
        self.__last_obs_time = time.time()
        self._observe()
        for robot in self.robots.values():
            robot.arm.on_reset_end()

    def terminate_episode(self):
        for robot in self.robots.values():
            robot.arm.on_episode_end()

    def shutdown(self):
        if self.__optitrack is not None:
            self.__optitrack.shutdown()
        for robot in self.robots.values():
            robot.gripper.shutdown()
            robot.arm.shutdown()
        super(RealEnvironment, self).shutdown()

    def _observe(self):
        for robot in self.robots.values():
            robot.gripper.start_observing()
            robot.arm.start_observing()
        if self.__optitrack is not None:
            self.__optitrack.update()
        for robot in self.robots.values():
            robot.gripper.finish_observing()
            robot.arm.finish_observing()

    def _step(self):
        act_time = None
        if self.synchronized_mode:
            assert self.__last_obs_time is not None, \
                "Environment.reset() must be called before stepping if Environment.synchronized is set."
            act_time = self.__last_obs_time + self.__obs_act_offset_s
            sleep_time = act_time - time.time()
            if sleep_time < 0:
                act_time = time.time()  # do not carry the time debt into the next step
                if self.__fail_on_timing_missed:
                    raise TimingMissedError("action", -sleep_time)
                else:
                    logger.warning("Missed action timing by {:0.2f}ms.".format(-sleep_time * 1000))
            else:
                time.sleep(sleep_time)
        for robot in self.robots.values():
            robot.gripper.start_acting()
            robot.arm.start_acting()
        for robot in self.robots.values():
            robot.gripper.finish_acting()
            robot.arm.finish_acting()
        if self.synchronized_mode:
            self.__last_obs_time = act_time - self.__obs_act_offset_s + self.time_step
            sleep_time = self.__last_obs_time - time.time()
            if sleep_time < 0:
                self.__last_obs_time = time.time()  # do not carry the time debt into the next step
                if self.__fail_on_timing_missed:
                    raise TimingMissedError("observation", -sleep_time)
                else:
                    logger.warning("Missed observation timing by {:0.2f}ms.".format(-sleep_time * 1000))
            else:
                time.sleep(sleep_time)
        self._observe()

    @property
    def optitrack(self):
        return self.__optitrack
