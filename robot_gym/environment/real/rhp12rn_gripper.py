import time
from threading import Lock
from typing import Tuple, Optional, List, Dict, Any, Union

import numpy as np
from rhp12rn import RHP12RNConnector, FieldReadFuture, FieldWriteFuture, RHP12RNAConnector, DynamixelCommunicationError

from robot_gym.environment import RobotComponent, JointMode
from .synchronized_observer import SynchronizedObserver
from transformation import Transformation
from .ur10_robot_arm import UR10RobotArm


class RHP12RNGripper(RobotComponent, SynchronizedObserver):
    def __init__(self, connector: Union[RHP12RNConnector, RHP12RNAConnector], async_actions: bool = False):
        RobotComponent.__init__(self)
        SynchronizedObserver.__init__(self)
        self.__connector = connector
        self.__pulses_per_rad = 660 / 1.1  # TODO: this is probably very inaccurate
        self.__vel_factor = (100 * 60) / 2 * np.pi
        self.__observed_fields = [
            "goal_velocity",
            "goal_position",
            "present_position",
            "present_velocity",
            "velocity_limit",
            "min_position_limit",
            "max_position_limit"
        ]
        self.__read_values: Optional[Dict[str, Any]] = None
        self.__write_values: Dict[str, Any] = {}

        self.__read_futures: Optional[List[FieldReadFuture]] = None
        self.__write_futures: Optional[List[FieldWriteFuture]] = None

        self.__attached_arm: Optional[UR10RobotArm] = None
        self.__lock = Lock()
        self.__async_actions = async_actions
        self.observations_enabled = True

    def attach_to(self, arm: UR10RobotArm):
        self.__attached_arm = arm

    def start_observing(self):
        if self.observations_enabled:
            with self.__lock:
                self.__read_futures = [self.__connector.read_field_async(f) for f in self.__observed_fields]

    def finish_observing(self):
        if self.observations_enabled:
            with self.__lock:
                self.__read_values = {
                    field_name: future.result()
                    for field_name, future in zip(self.__observed_fields, self.__read_futures)
                }

    def start_acting(self):
        with self.__lock:
            self.__write_futures = [
                self.__connector.write_field_async(name, value)
                for name, value in self.__write_values.items()]
            self.__write_values.clear()

    def finish_acting(self):
        if not self.__async_actions:
            with self.__lock:
                for f in self.__write_futures:
                    f.result()

    def initialize(self):
        with self.__lock:
            self.__connector.connect()
            self.__connector.write_field("torque_enable", False)
            self.__connector.write_field("operating_mode", 5)
            self.__connector.write_field("torque_enable", True)

    def shutdown(self):
        with self.__lock:
            try:
                self.__connector.write_field("torque_enable", False)
            except DynamixelCommunicationError:
                self.__reconnect_sync()
                self.__connector.write_field("torque_enable", False)
            finally:
                self.__connector.disconnect()

    def reconnect(self):
        with self.__lock:
            self.__reconnect_sync()

    def __reconnect_sync(self):
        self.__connector.disconnect()
        time.sleep(0.5)
        self.__connector.connect()

    def get_joint_target_velocities(self) -> np.ndarray:
        return np.array([self.__read_values["goal_velocity"] / self.__vel_factor])

    def set_joint_target_velocities(self, velocities: np.ndarray):
        self.__write_values["goal_velocity"] = int(round(velocities[0] * self.__vel_factor))

    def get_joint_target_positions(self) -> np.ndarray:
        return np.array([self.__read_values["goal_position"] / self.__pulses_per_rad])

    def set_joint_target_positions(self, positions: np.ndarray):
        self.__write_values["goal_position"] = int(round(positions[0] * self.__pulses_per_rad))

    def set_joint_torques(self, torques: np.ndarray):
        raise NotImplementedError()

    def set_joint_mode(self, mode: JointMode):
        if mode is not JointMode.POSITION_CONTROL:
            raise NotImplementedError()

    def get_joint_mode(self) -> JointMode:
        return JointMode.POSITION_CONTROL

    def get_joint_positions(self) -> np.ndarray:
        return np.array([self.__read_values["present_position"] / self.__pulses_per_rad])

    def get_joint_velocities(self) -> np.ndarray:
        return np.array([self.__read_values["present_velocity"] / self.__vel_factor])

    def get_joint_velocity_limits(self) -> np.ndarray:
        return np.array([self.__read_values["velocity_limit"] / self.__vel_factor])

    def get_joint_intervals(self) -> np.ndarray:
        return np.array([[self.__read_values["min_position_limit"],
                          self.__read_values["max_position_limit"]]]) / self.__pulses_per_rad

    def get_pose(self) -> Transformation:
        if self.__attached_arm is None:
            # Assume world frame is gripper base
            return Transformation()
        else:
            return self.__attached_arm.pose.transform(self.__attached_arm.tcp_pose_robot_frame)

    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.__attached_arm is None:
            return np.zeros(3), np.zeros(3)
        else:
            return tuple(self.__attached_arm.pose.rotation.apply(self.__attached_arm.tcp_velocity_robot_frame))
