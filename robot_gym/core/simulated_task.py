import time
import traceback
from abc import ABC
from typing import Iterable, TYPE_CHECKING, Optional

from robot_gym import logger
from .base_task import BaseTask
from robot_gym.environment.simulation import SimulationEnvironment

if TYPE_CHECKING:
    from .controllers import Controller
    from .sensors import Sensor
    from .rewards import Reward


class SimulatedTask(BaseTask[SimulationEnvironment], ABC):
    """
    An abstract base class for gym environments that use the simulated robot.
    The task-specific aspects of the environment are defined by overriding the *_task methods.
    """

    def __init__(self, controllers: Iterable["Controller[BaseTask]"],
                 sensors: Iterable["Sensor[BaseTask]"], rewards: Iterable["Reward[BaseTask]"],
                 time_step: float = 0.005, time_limit_steps: Optional[int] = None):
        super().__init__(controllers, sensors, rewards, time_step, time_limit_steps)

    def _initialize(self):
        self._initialize_scene()
        self.environment.set_reset_checkpoint()

    def _initialize_scene(self):
        pass

    def _restart_env(self):
        """
        Deals with CoppeliaSim's bullshit.
        :return:
        """
        logger.info("Terminating Simulator...")
        self.environment.shutdown()
        logger.info("Simulator terminated.")
        num_restart_attempts = 10
        for i in range(10):
            logger.info("Trying to restart Simulator for the {}/{}. time...".format(i + 1, num_restart_attempts))
            try:
                self.environment.initialize(self.time_step)
                self._initialize()
                logger.info("Success!")
                return
            except KeyboardInterrupt:
                raise
            except:
                logger.error("Failed with the following error message:")
                logger.error(traceback.format_exc())
                logger.info("Retrying in 10s...")
                time.sleep(10)
        logger.error("Giving up :(")
        raise RuntimeError("Failed to restart Simulator")
