import time
from threading import Lock
from typing import Tuple, Optional, Sequence, Literal

import numpy as np
from dashboard_client import DashboardClient
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from scipy.spatial.transform import Rotation

from robot_gym.environment import RobotComponent, JointMode, RobotArm
from robot_gym import logger
from .optitrack import Optitrack
from .hand_eye_calibration import estimate_hand_eye_calibration, compute_pose_error, HandEyeCalibrationResult
from .synchronized_actor import synchronized_act, SimpleSynchronizedActor
from .synchronized_observer import SimpleSynchronizedObserver, synchronized_obs
from transformation import Transformation
from .util import solve_tsp_two_opt


class UR10ConnectionLostException(Exception):
    def __init__(self, interface: Literal["receive", "control"]):
        super(UR10ConnectionLostException, self).__init__(
            "{} interface lost connection to the UR10.".format(interface.capitalize()))


class UR10ReconnectFailedException(Exception):
    def __init__(self, interface: Literal["receive", "control"], trials: int):
        super(UR10ReconnectFailedException, self).__init__(
            "Failed to reestablish connection to {} interface after {} trials.".format(interface.capitalize(), trials))


class UR10RobotArm(RobotArm, SimpleSynchronizedObserver, SimpleSynchronizedActor):
    _ROBOT_MODES = {
        -1: "ROBOT_MODE_NO_CONTROLLER",
        0: "ROBOT_MODE_DISCONNECTED",
        1: "ROBOT_MODE_CONFIRM_SAFETY",
        2: "ROBOT_MODE_BOOTING",
        3: "ROBOT_MODE_POWER_OFF",
        4: "ROBOT_MODE_POWER_ON",
        5: "ROBOT_MODE_IDLE",
        6: "ROBOT_MODE_BACKDRIVE",
        7: "ROBOT_MODE_RUNNING",
        8: "ROBOT_MODE_UPDATING_FIRMWARE"
    }

    def __init__(self, robot_address: str, ur_cap_port: int = 50002,
                 robot_pose: Optional[Transformation] = None, joint_velocity_limits: Optional[np.ndarray] = None,
                 optitrack_tcp_name: str = "tcp", calibration_safe_box_extents: Sequence[float] = (0.3, 0.3, 0.3),
                 calibration_safe_box_pose: Optional[Transformation] = None,
                 calibration_safe_orientation_center: Optional[Rotation] = None,
                 calibration_safe_rotation_angle: float = 0.5,
                 calibration_velocity: float = 0.3, watchdog_timeout_reset: float = 10.0,
                 watchdog_timeout_execution: float = 0.5, watchdog_timeout_idle: float = 120.0,
                 joint_target_acceleration: float = 1.4, joint_target_velocity: float = 0.3,
                 linear_target_acceleration: float = 1.2, angular_target_acceleration: float = 1.2,
                 fixed_hand_eye_calibration: Optional[HandEyeCalibrationResult] = None):
        self.__robot_lock = Lock()
        SimpleSynchronizedObserver.__init__(self, self.__robot_lock)
        SimpleSynchronizedActor.__init__(self, self.__robot_lock, action_protocol_length=1000)
        RobotComponent.__init__(self)
        self.__robot_hostname = robot_address
        self.__ur_cap_port = ur_cap_port
        self.__control_interface: Optional[RTDEControlInterface] = None
        self.__receive_interface: Optional[RTDEReceiveInterface] = None
        self.__robot_pose = robot_pose if robot_pose is not None else Transformation()
        self.__joint_target_acceleration = joint_target_acceleration
        self.__joint_target_velocity = joint_target_velocity
        self.__linear_target_acceleration = linear_target_acceleration
        self.__angular_target_acceleration = angular_target_acceleration
        self.__marker_pose_tcp_frame = None
        if joint_velocity_limits is not None:
            assert joint_velocity_limits.shape == (6,)
            self.__joint_velocity_limits = joint_velocity_limits
        else:
            self.__joint_velocity_limits = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
        self.__joint_mode = JointMode.POSITION_CONTROL
        self.__optitrack_tcp_name = optitrack_tcp_name
        self.__calibration_safe_box_extents = np.array(calibration_safe_box_extents)
        self.__calibration_safe_box_pose = calibration_safe_box_pose
        self.__calibration_safe_orientation_center = calibration_safe_orientation_center
        self.__calibration_safe_rotation_angle = calibration_safe_rotation_angle
        self.__calibration_velocity = calibration_velocity
        self.__hand_eye_calibration_result = fixed_hand_eye_calibration
        self.__hand_eye_calibration_is_fixed = fixed_hand_eye_calibration is not None
        self.__watchdog_timeout_reset = watchdog_timeout_reset
        self.__watchdog_timeout_idle = watchdog_timeout_idle
        self.__watchdog_timeout_execution = watchdog_timeout_execution
        self.__current_mode = None
        self.__robot_is_in_protective_stop = False

    def initialize(self, optitrack: Optional[Optitrack] = None):
        self.__receive_interface = RTDEReceiveInterface(self.__robot_hostname)
        dashboard_client = DashboardClient(self.__robot_hostname)
        dashboard_client.connect()
        if dashboard_client.safetystatus() == "Safetystatus: FAULT":
            print("Robot is in FAULT state. Restarting safety...")
            dashboard_client.restartSafety()
            while dashboard_client.safetystatus() == "Safetystatus: FAULT":
                time.sleep(0.2)
            print("Robot state changed to {}.".format(dashboard_client.safetystatus().split(":")[1].strip()))
            time.sleep(1.0)
        if dashboard_client.safetystatus() == "Safetystatus: PROTECTIVE_STOP":
            print("Robot is in PROTECTIVE_STOP state. Unlocking protective stop...")
            dashboard_client.unlockProtectiveStop()
            while dashboard_client.safetystatus() == "Safetystatus: PROTECTIVE_STOP":
                time.sleep(0.2)
            print("Robot state changed to {}.".format(dashboard_client.safetystatus().split(":")[1].strip()))
            time.sleep(1.0)
        self.__robot_is_in_protective_stop = False
        if self.__get_robot_mode() in ["ROBOT_MODE_POWER_OFF", "ROBOT_MODE_IDLE", "ROBOT_MODE_BOOTING"]:
            if self.__get_robot_mode() == "ROBOT_MODE_BOOTING":
                logger.info("Waiting for the robot to boot...")
                while self.__get_robot_mode() == "ROBOT_MODE_BOOTING":
                    time.sleep(0.5)
            if self.__get_robot_mode() == "ROBOT_MODE_POWER_OFF":
                dashboard_client.powerOn()
                logger.info("Waiting for the robot to power on...")
                while self.__get_robot_mode() != "ROBOT_MODE_IDLE":
                    time.sleep(0.5)
                logger.info("Robot is powered on.")
            if self.__get_robot_mode() == "ROBOT_MODE_IDLE":
                dashboard_client.brakeRelease()
                logger.info("Waiting for the brakes to release...")
                while self.__get_robot_mode() != "ROBOT_MODE_RUNNING":
                    time.sleep(0.5)
                logger.info("Brakes are released.")
        dashboard_client.disconnect()
        self.__control_interface = RTDEControlInterface(self.__robot_hostname, ur_cap_port=self.__ur_cap_port)
        if not self.__control_interface.isConnected():
            dashboard_client.disconnect()
            raise UR10ConnectionLostException("control")
        if not self.__receive_interface.isConnected():
            dashboard_client.disconnect()
            raise UR10ConnectionLostException("receive")
        if not self.__hand_eye_calibration_is_fixed and optitrack is not None:
            self.__hand_eye_calibration_result = self.run_hand_eye_calibration(optitrack)
        elif optitrack is None:
            logger.info("No Optitrack specified. Assuming robot frame is world frame.")

        if self.__hand_eye_calibration_result is not None:
            mean_ang_err_deg = self.__hand_eye_calibration_result.mean_rotational_error / np.pi * 180
            assert self.__hand_eye_calibration_result.mean_translational_error < 5e-3, \
                "Mean linear pose error too large"
            assert mean_ang_err_deg < 1, "Mean angular pose error too large"
            self.__robot_pose = self.__hand_eye_calibration_result.robot_pose_world_frame
            self.__marker_pose_tcp_frame = self.__hand_eye_calibration_result.marker_pose_tcp_frame

    def run_hand_eye_calibration(self, optitrack: Optitrack, num_samples: int = 200,
                                 fixed_marker_pose_tcp_frame: Optional[Transformation] = None):
        optitrack_tcp = optitrack.rigid_bodies[self.__optitrack_tcp_name]
        generator = np.random.default_rng(0)
        tcp_pose = self.__from_ur_pose(self.__receive_interface.getActualTCPPose())
        if self.__calibration_safe_box_pose is None:
            # Use current position as center and no rotation
            self.__calibration_safe_box_pose = Transformation(tcp_pose.translation)
        if self.__calibration_safe_orientation_center is None:
            self.__calibration_safe_orientation_center = tcp_pose.rotation
        center_pose = Transformation(self.__calibration_safe_box_pose.translation,
                                     self.__calibration_safe_orientation_center)
        tcp_target_poses = []
        for i in range(num_samples):
            translation_box_frame = (generator.random(3) - 0.5) * self.__calibration_safe_box_extents
            translation_robot_frame = self.__calibration_safe_box_pose.transform(translation_box_frame)
            rotation_direction = generator.random(3)
            rotation_amount = generator.random(1) * self.__calibration_safe_rotation_angle
            rotation_rotvec_center_frame = rotation_direction * rotation_amount
            rotation_robot_frame = self.__calibration_safe_orientation_center * \
                                   Rotation.from_rotvec(rotation_rotvec_center_frame)
            target_tcp_pose_robot_frame = Transformation(translation_robot_frame, rotation_robot_frame)
            tcp_target_poses.append(target_tcp_pose_robot_frame)

        # Try to find an order that minimizes distance by approximately solving the TCP
        logger.info("Optimizing pose order...")
        positions = np.array([pose.translation for pose in tcp_target_poses])
        order = solve_tsp_two_opt(positions)
        tcp_target_poses = [tcp_target_poses[i] for i in order]

        # Reach initial pose
        lin_error, ang_error = compute_pose_error(tcp_pose, center_pose)
        if lin_error > 0 or ang_error > 0:
            while lin_error >= 0.4 or ang_error >= np.pi:
                input("The robot is too far from the starting location. Please move it manually to coordinates "
                      "pos=({:0.2f}, {:0.2f}, {:0.2f}), rot=({:0.2f}, {:0.2f}, {:0.2f}) and press enter...".format(
                    *center_pose.translation, *center_pose.rotvec))
                tcp_pose = self.__from_ur_pose(self.__receive_interface.getActualTCPPose())
                lin_error, ang_error = compute_pose_error(tcp_pose, center_pose)
            input(
                "The robot will now move to the center location of the calibration box. Press enter to continue...")
            self.move_to_pose(center_pose, velocity=self.__calibration_velocity)

        input("The robot will now move randomly in a box of size ({}, {}, {}). Confirm "
              "that there is space and press enter...".format(*self.__calibration_safe_box_extents))
        tcp_poses_world_frame = []
        tcp_poses_robot_frame = []
        for i, pose in enumerate(tcp_target_poses):
            logger.info("Moving to pose {}/{}...".format(i + 1, num_samples))
            self.move_to_pose(pose, velocity=self.__calibration_velocity)
            time.sleep(0.2)
            optitrack.update()
            assert optitrack_tcp.tracking_valid, "TCP is not tracked by Optitrack. Make sure TCP is in sight of " \
                                                 "at least 3 Optitrack cameras."
            optitrack_tcp_pose = optitrack_tcp.pose
            tcp_poses_world_frame.append(optitrack_tcp_pose)
            tcp_pose = self.__from_ur_pose(self.__receive_interface.getActualTCPPose())
            tcp_poses_robot_frame.append(tcp_pose)
        logger.info("Moving to pose center pose...")
        self.move_to_pose(center_pose, velocity=self.__calibration_velocity)
        logger.info("Done collecting samples." + " " * 30)

        # Figure out transformation between optitrack gripper body and robot tcp, as well as the global robot pose
        logger.info("Optimizing hand-eye calibration...")
        hand_eye_calibration_result = estimate_hand_eye_calibration(
            tcp_poses_robot_frame, tcp_poses_world_frame, fixed_marker_pose_tcp_frame)
        mean_ang_err_deg = hand_eye_calibration_result.mean_rotational_error / np.pi * 180
        logger.info("Mean linear pose error: {:0.2f}mm".format(
            hand_eye_calibration_result.mean_translational_error * 1000))
        logger.info("Mean angular pose error: {:0.4f} deg".format(mean_ang_err_deg))
        return hand_eye_calibration_result

    def shutdown(self):
        with self.__robot_lock:
            try:
                # Check whether robot is not already in protective stop or similar
                if self.__receive_interface.getSafetyStatusBits() & 0x3F4 == 0:
                    self.__stop_synchronous()
            except UR10ConnectionLostException:
                self.reconnect()
                self.__stop_synchronous()
            finally:
                try:
                    self.__control_interface.disconnect()
                finally:
                    self.__receive_interface.disconnect()

    @synchronized_obs
    def get_joint_target_velocities(self) -> np.ndarray:
        return np.array(self.__receive_interface.getActualQd())

    @synchronized_act
    def set_joint_target_velocities(self, velocities: np.ndarray):
        assert self.__joint_mode in [JointMode.POSITION_CONTROL, JointMode.VELOCITY_CONTROL]
        self.__check_state()
        if self.__joint_mode == JointMode.POSITION_CONTROL:
            assert np.all(velocities == velocities[0]), "Only a single velocity is supported."
            self.__joint_target_velocity = velocities[0]
        else:
            velocities = np.clip(velocities, -self.__joint_velocity_limits, self.__joint_velocity_limits)
            self.__current_mode = "speedj"
            self.__check_call(self.__control_interface.speedJ(velocities, self.__joint_target_acceleration, 0.001))

    @synchronized_obs
    def get_joint_target_positions(self) -> np.ndarray:
        return np.array(self.__receive_interface.getTargetQ())

    @synchronized_act
    def set_joint_target_positions(self, positions: np.ndarray):
        self.__check_state()
        assert self.__joint_mode == JointMode.POSITION_CONTROL
        self.__current_mode = "movej"
        self.__check_call(self.__control_interface.moveJ(
            positions.tolist(), self.__joint_target_velocity, self.__joint_target_acceleration, True))

    @synchronized_act
    def set_joint_torques(self, torques: np.ndarray):
        raise NotImplementedError()

    def __stop_synchronous(self):
        self.__check_state()
        if self.__current_mode == "movej":
            self.__check_call(self.__control_interface.stopJ(self.__joint_target_acceleration))
        elif self.__current_mode == "movel":
            self.__check_call(self.__control_interface.stopL(self.__linear_target_acceleration))
        elif self.__current_mode == "speedt":
            self.__check_call(self.__control_interface.speedStop2(
                self.__linear_target_acceleration, self.__angular_target_acceleration))
        elif self.__current_mode == "speedl":
            self.__check_call(self.__control_interface.speedStop(self.__linear_target_acceleration))
        else:
            self.__check_call(self.__control_interface.speedStop(self.__joint_target_acceleration))
        self.__current_mode = None

    def stop(self):
        with self.__robot_lock:
            self.__stop_synchronous()

    def set_joint_mode(self, mode: JointMode):
        self.stop()
        self.__joint_mode = mode

    def get_joint_mode(self) -> JointMode:
        return self.__joint_mode

    @synchronized_obs
    def get_joint_positions(self) -> np.ndarray:
        return np.array(self.__receive_interface.getActualQ())

    @synchronized_obs
    def get_joint_velocities(self) -> np.ndarray:
        return np.array(self.__receive_interface.getActualQd())

    def get_joint_velocity_limits(self) -> np.ndarray:
        return self.__joint_velocity_limits

    def get_joint_intervals(self) -> np.ndarray:
        return np.array([[-np.inf] * 6, [np.inf] * 6]).T

    def get_pose(self) -> Transformation:
        return self.__robot_pose

    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.zeros(3)

    @synchronized_obs
    def get_end_effector_to_base_transform(self) -> Transformation:
        pose_raw = self.__receive_interface.getActualTCPPose()
        translation = pose_raw[:3]
        rotation = Rotation.from_rotvec(pose_raw[3:])
        return Transformation(translation, rotation)

    @property
    def end_effector_pose(self) -> Transformation:
        return self.__robot_pose.transform(self.get_end_effector_to_base_transform())

    def move_to_pose(self, target_pose_world_frame: Transformation, velocity: float = 0.25):
        with self.__robot_lock:
            self.__check_state()
            target_pose_robot_frame = self.__robot_pose.transform(target_pose_world_frame, inverse=True)
            if self.__current_mode == "speedl":
                self.__stop_synchronous()
            self.__check_call(self.__control_interface.moveL(
                self.__to_ur_pose(target_pose_robot_frame), velocity, self.__linear_target_acceleration))

    @synchronized_act
    def move_towards_pose_linear(
            self, target_pose_world_frame: Transformation, linear_velocity: float = 0.25,
            angular_velocity: Optional[float] = None, linear_target_acceleration: Optional[float] = None,
            angular_target_acceleration: Optional[float] = None):
        self.__check_state()
        target_pose_robot_frame = self.__robot_pose.transform(target_pose_world_frame, inverse=True)
        if linear_target_acceleration is None:
            linear_target_acceleration = self.__linear_target_acceleration
        if angular_target_acceleration is None:
            angular_target_acceleration = self.__angular_target_acceleration
        if angular_velocity is None:
            if self.__current_mode != "movel":
                self.__stop_synchronous()
            self.__check_call(self.__control_interface.moveL(
                self.__to_ur_pose(target_pose_robot_frame), linear_velocity, linear_target_acceleration, True))
            self.__current_mode = "movel"
        else:
            # In this mode, the target rotation and the target translation might be reached at different times
            if self.__current_mode != "speedt":
                self.__stop_synchronous()
            self.__check_call(self.__control_interface.speedT(
                self.__to_ur_pose(target_pose_robot_frame), linear_velocity, angular_velocity,
                linear_target_acceleration, angular_target_acceleration))
            self.__current_mode = "speedt"

    @synchronized_act
    def move_cartesian_velocity(self, linear_vel_world_frame: np.ndarray, angular_vel_world_frame: np.ndarray):
        linear_vel_robot_frame, angular_vel_world_frame = self.__robot_pose.rotation.apply(
            [linear_vel_world_frame, angular_vel_world_frame], inverse=True)
        self.__check_state()
        target_vel = np.concatenate([linear_vel_robot_frame, angular_vel_world_frame])
        self.__check_call(
            self.__control_interface.speedL(target_vel.tolist(), self.__linear_target_acceleration, 0.001))
        self.__current_mode = "speedl"

    @synchronized_obs
    def get_tcp_pose_robot_frame(self) -> Transformation:
        return self.__from_ur_pose(self.__receive_interface.getActualTCPPose())

    @synchronized_obs
    def get_tcp_velocity_robot_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        ur_speed = self.__receive_interface.getActualTCPSpeed()
        return ur_speed[:3], ur_speed[3:]

    @property
    def tcp_pose_robot_frame(self):
        return self.get_tcp_pose_robot_frame()

    @property
    def tcp_velocity_robot_frame(self):
        return self.get_tcp_velocity_robot_frame()

    @property
    def hand_eye_calibration_result(self) -> Optional[HandEyeCalibrationResult]:
        return self.__hand_eye_calibration_result

    def start_acting(self):
        with self.__robot_lock:
            self.__check_state()
        super(UR10RobotArm, self).start_acting()

    def start_observing(self):
        with self.__robot_lock:
            if not self.__receive_interface.isConnected():
                raise UR10ConnectionLostException("receive")
        super(UR10RobotArm, self).start_observing()

    def __get_robot_mode(self) -> str:
        rm = self.__receive_interface.getRobotMode()
        if rm not in self._ROBOT_MODES:
            raise UR10ConnectionLostException("receive")
        return self._ROBOT_MODES[rm]

    def __check_state(self):
        assert not self.__robot_is_in_protective_stop, "Robot is in protective stop."
        robot_mode = self.__get_robot_mode()
        assert robot_mode == "ROBOT_MODE_RUNNING", "Unexpected robot mode {}".format(robot_mode)
        self.__check_call(self.__control_interface.kickWatchdog())

    def on_reset_start(self):
        self.__check_state()
        self.__check_call(self.__control_interface.setWatchdog(1 / self.__watchdog_timeout_reset))

    def on_reset_end(self):
        self.__check_state()
        self.__check_call(self.__control_interface.setWatchdog(1 / self.__watchdog_timeout_execution))

    def on_episode_start(self):
        self.__check_state()
        self.__check_call(self.__control_interface.setWatchdog(1 / self.__watchdog_timeout_idle))

    def on_episode_end(self):
        self.__check_call(self.__control_interface.kickWatchdog())
        if not self.__robot_is_in_protective_stop:
            self.stop()
        self.__check_call(self.__control_interface.setWatchdog(1 / self.__watchdog_timeout_idle))

    @staticmethod
    def __to_ur_pose(pose: Transformation):
        return np.concatenate([pose.translation, pose.rotvec]).tolist()

    @staticmethod
    def __from_ur_pose(ur_pose: Sequence[float]):
        return Transformation.from_pos_rotvec(ur_pose[:3], ur_pose[3:])

    @staticmethod
    def __check_call(call_output: bool, interface: Literal["control", "receive"] = "control"):
        if not call_output:
            raise UR10ConnectionLostException(interface)

    def stop_linear_control(self):
        with self.__robot_lock:
            self.__stop_synchronous()

    def protective_stop(self):
        with self.__robot_lock:
            self.__robot_is_in_protective_stop = True
            try:
                self.__check_call(self.__control_interface.triggerProtectiveStop())
            except UR10ConnectionLostException:
                self.reconnect()
                self.__check_call(self.__control_interface.triggerProtectiveStop())

    def reconnect(self, force_disconnect_control: bool = False, force_disconnect_receive: bool = False):
        with self.__robot_lock:
            if not self.__robot_is_in_protective_stop:
                if force_disconnect_control:
                    self.__control_interface.disconnect()
                if force_disconnect_receive:
                    self.__receive_interface.disconnect()
                if force_disconnect_control or force_disconnect_receive:
                    time.sleep(0.5)
                control_attempts = 0
                receive_attempts = 0
                last_connect_attempt = time.time()
                connection_ok = self.__control_interface.isConnected() and self.__control_interface.isProgramRunning() \
                                and self.__receive_interface.isConnected()
                while not connection_ok and control_attempts <= 10 and receive_attempts <= 10:
                    if not self.__control_interface.isConnected() or not self.__control_interface.isProgramRunning():
                        # Immediately reconnect to control interface due to the watchdog
                        self.__control_interface.disconnect()
                        # For some reason reconnect simply hangs here
                        self.__control_interface = RTDEControlInterface(
                            self.__robot_hostname, ur_cap_port=self.__ur_cap_port)
                        control_attempts += 1
                    else:
                        self.__control_interface.kickWatchdog()
                    if not self.__receive_interface.isConnected():
                        interval = 0.5 * 2 ** receive_attempts
                        if time.time() - last_connect_attempt >= interval:
                            self.__receive_interface = RTDEReceiveInterface(self.__robot_hostname)
                            last_connect_attempt = time.time()
                            receive_attempts += 1
                    connection_ok = self.__control_interface.isConnected() and self.__receive_interface.isConnected()
                    if not connection_ok:
                        time.sleep(0.3)
                if not self.__control_interface.isConnected() or not self.__control_interface.isProgramRunning():
                    raise UR10ReconnectFailedException("control", control_attempts)
                if not self.__receive_interface.isConnected():
                    raise UR10ReconnectFailedException("receive", receive_attempts)

    def shutdown_robot(self):
        with self.__robot_lock:
            dashboard_client = DashboardClient(self.__robot_hostname)
            dashboard_client.connect()
            dashboard_client.shutdown()
            dashboard_client.disconnect()

    def power_off(self):
        with self.__robot_lock:
            dashboard_client = DashboardClient(self.__robot_hostname)
            dashboard_client.connect()
            dashboard_client.powerOff()
            dashboard_client.disconnect()

    def check_robot_alive(self):
        with self.__robot_lock:
            dashboard_client = DashboardClient(self.__robot_hostname)
            try:
                dashboard_client.connect()
                dashboard_client.disconnect()
                return True
            except RuntimeError:
                return False

    def __repr__(self):
        return "UR10RobotArm({}:{})".format(self.__robot_hostname, self.__ur_cap_port)
