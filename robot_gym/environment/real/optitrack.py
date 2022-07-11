import time
from typing import Optional, Tuple, Dict

import numpy as np

from natnet_client import NatNetClient, DataFrame, DataDescriptions, RigidBodyDescription, RigidBody, LabeledMarker

from robot_gym import logger
from robot_gym.environment import Object
from robot_gym.util import ReadOnlyDict
from transformation import Transformation
from util.event import Event


class _OptitrackRigidBody:
    def __init__(self, rigid_body_desc: RigidBodyDescription, world_frame: Transformation):
        self.__rigid_body_desc = rigid_body_desc
        self.__pose_estimate: Optional[Transformation] = None
        self.__velocity_estimate: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.__prev_time_stamp = None
        self.__tracking_valid = False
        self.__world_frame = world_frame
        self.__last_update_time = None

    def add_frame(self, frame: RigidBody, time_stamp: float):
        # TODO: Kalman filter and test this
        if frame.tracking_valid:
            new_pose = Transformation.from_pos_quat(position=np.array(frame.pos), quaternion=np.array(frame.rot))
            new_pose_world_frame = self.__world_frame.transform(new_pose)
            if self.__pose_estimate is None:
                self.__velocity_estimate = (np.zeros(3), np.zeros(3))
            else:
                dt = time_stamp - self.__prev_time_stamp
                lin_vel = (new_pose_world_frame.translation - self.__pose_estimate.translation) / dt
                ang_vel = (self.__pose_estimate.rotation.inv() * new_pose_world_frame.rotation).as_euler("xyz") / dt
                self.__velocity_estimate = (lin_vel, ang_vel)
            self.__pose_estimate = new_pose_world_frame
            self.__prev_time_stamp = time_stamp
        self.__tracking_valid = frame.tracking_valid
        self.__last_update_time = time.time()

    @property
    def rigid_body_desc(self) -> RigidBodyDescription:
        return self.__rigid_body_desc

    @property
    def pose(self) -> Optional[Transformation]:
        return self.__pose_estimate

    @property
    def velocity(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self.__velocity_estimate

    @property
    def tracking_valid(self):
        return self.__tracking_valid

    @property
    def last_update_time(self) -> Optional[float]:
        return self.__last_update_time


class OptitrackRigidBody(Object):
    def __init__(self, inner_object: _OptitrackRigidBody):
        self.__inner_object = inner_object
        self.__pose_estimate: Optional[Transformation] = None
        self.__velocity_estimate: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.__tracking_valid = False
        self.__last_update_time = None

    def _update_state(self):
        self.__pose_estimate = self.__inner_object.pose
        self.__velocity_estimate = self.__inner_object.velocity
        self.__tracking_valid = self.__inner_object.tracking_valid
        self.__last_update_time = self.__inner_object.last_update_time

    def get_pose(self) -> Transformation:
        assert self.__pose_estimate is not None
        return self.__pose_estimate

    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.__velocity_estimate is not None
        return self.__velocity_estimate

    @property
    def tracking_valid(self):
        return self.__tracking_valid

    @property
    def description(self) -> RigidBodyDescription:
        return self.__inner_object.rigid_body_desc

    @property
    def last_update_time(self) -> Optional[float]:
        return self.__last_update_time


class Optitrack:
    def __init__(self, server_ip_address: str = "127.0.0.1", local_ip_address: str = "127.0.0.1",
                 multicast_address: str = "239.255.42.99", command_port: int = 1510, data_port: int = 1511,
                 use_multicast: bool = True, world_transformation: Optional[Transformation] = None):
        self.__nat_net_client = NatNetClient(server_ip_address, local_ip_address, multicast_address, command_port,
                                             data_port, use_multicast)
        self.__data_desc: Optional[DataDescriptions] = None
        self.__rigid_bodies_internal: Optional[Dict[int, _OptitrackRigidBody]] = None
        self.__markers: Optional[Tuple[LabeledMarker, ...]] = None
        self.__marker_update_time_stamp: Optional[float] = None
        self.__markers_internal: Optional[Tuple[LabeledMarker, ...]] = None
        self.__marker_update_timestamp_internal: Optional[float] = None
        self.__rigid_bodies: Optional[ReadOnlyDict[str, OptitrackRigidBody]] = None
        self.__world_frame = world_transformation if world_transformation is not None else Transformation()
        self.__on_data_frame_received_event = Event()

    def initialize(self):
        assert not self.initialized
        try:
            self.__nat_net_client.on_data_description_received_event.handlers.append(self.__on_data_desc)
            self.__nat_net_client.on_data_frame_received_event.handlers.append(self.__on_data_frame)
            self.__nat_net_client.connect()
            self.__nat_net_client.request_modeldef()
            logger.info("Waiting for model definitions...")
            start_time = time.time()
            while self.__data_desc is None and time.time() - start_time < 10.0:
                self.__nat_net_client.update_sync()
            assert self.__data_desc is not None, "Timed out waiting for data descriptions from Optitrack."
            logger.info("Received model definitions.")
            self.__rigid_bodies_internal = {
                rb.id_num: _OptitrackRigidBody(rb, self.__world_frame) for rb in self.__data_desc.rigid_bodies
            }
            self.__rigid_bodies = ReadOnlyDict(
                {rb.rigid_body_desc.name: OptitrackRigidBody(rb) for rb in self.__rigid_bodies_internal.values()})
            self.__markers_internal = []
        except:
            self.shutdown()
            raise

    def update(self):
        assert self.initialized
        self.__nat_net_client.update_sync()
        for rb in self.__rigid_bodies.values():
            rb._update_state()
        self.__marker_update_time_stamp = self.__marker_update_timestamp_internal
        self.__markers = tuple(
            LabeledMarker(m.id_num, tuple(self.__world_frame.transform(np.array(m.pos))), m.size, m.param,
                          m.residual)
            for m in self.__markers_internal)

    def shutdown(self):
        self.__nat_net_client.on_data_description_received_event.handlers.remove(self.__on_data_desc)
        self.__nat_net_client.on_data_frame_received_event.handlers.remove(self.__on_data_frame)
        if self.__nat_net_client.connected:
            self.__nat_net_client.shutdown()
        self.__rigid_bodies_internal = self.__data_desc = self.__rigid_bodies = None
        self.__on_data_frame_received_event.handlers.clear()

    @property
    def data_desc(self) -> Optional[DataDescriptions]:
        return self.__data_desc

    def __on_data_frame(self, data_frame: DataFrame):
        if self.initialized:
            for rb in data_frame.rigid_bodies:
                if rb.tracking_valid:
                    rigid_body = self.__rigid_bodies_internal.get(rb.id_num)
                    if rigid_body is not None:
                        rigid_body.add_frame(rb, data_frame.suffix.timestamp)
            self.__markers_internal = data_frame.labeled_markers
            self.__marker_update_timestamp_internal = data_frame.suffix.timestamp
        self.__on_data_frame_received_event(data_frame)

    def __on_data_desc(self, data_desc: DataDescriptions):
        assert self.__data_desc is None
        self.__data_desc = data_desc

    @property
    def initialized(self) -> bool:
        return self.__rigid_bodies is not None

    @property
    def markers(self) -> Optional[Tuple[LabeledMarker, ...]]:
        return self.__markers

    @property
    def marker_update_timestamp(self) -> Optional[float]:
        return self.__marker_update_time_stamp

    @property
    def rigid_bodies(self) -> ReadOnlyDict[str, OptitrackRigidBody]:
        return self.__rigid_bodies

    @property
    def on_data_frame_received_event(self) -> Event:
        return self.__on_data_frame_received_event

    @property
    def optitrack_to_world_transformation(self) -> Transformation:
        return self.__world_frame
