from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TypeVar, Optional, Generic, Sequence, List, Union
import numpy as np
from trimesh import Trimesh

from .simulation_object import SimulationObject
from .simulation_robot import SimulationRobot
from robot_gym.environment.generic import Environment
from robot_gym.environment.generic.environment import ensure_initialized
from robot_gym import logger

ObjectType = TypeVar("ObjectType", bound=SimulationObject)
RobotType = TypeVar("RobotType", bound=SimulationRobot)
VisualShapeType = TypeVar("VisualShapeType")
CollisionShapeType = TypeVar("CollisionShapeType")


class ShapeTypes(Enum):
    MESH = 0
    BOX = 1
    SPHERE = 2


def ensure_not_initialized(func):
    def wrapped(self, *args, **kwargs):
        assert not self.initialized, "{} can only be called before the environment has been initialized".format(
            func.__name__)
        return func(self, *args, **kwargs)

    return wrapped


class SimulationEnvironment(Environment[ObjectType, RobotType], ABC,
                            Generic[ObjectType, RobotType, VisualShapeType, CollisionShapeType]):
    def __init__(self, display_name: str, simulator_step_s: float = 0.005, headless: bool = True):
        super(SimulationEnvironment, self).__init__(display_name)
        self.__robot = None
        self._substeps_per_step = None
        self._simulator_step_s = simulator_step_s
        self.__headless = headless
        self.__post_reset_robot_names: Optional[List[RobotType]] = None

    def _initialize(self, time_step: float):
        substeps_per_step_float = time_step / self._simulator_step_s
        self.__post_reset_robot_names = []
        self._substeps_per_step = int(round(substeps_per_step_float))
        actual_control_time_step = self._substeps_per_step * self._simulator_step_s
        if abs(actual_control_time_step / time_step - 1) > 0.01:
            logger.warning(
                "Control time step of {:.5f}s is not a multiple of simulator time step of {:.5f}s. "
                "The actual control time step will now be {:.5f}s.".format(
                    time_step, self._simulator_step_s, actual_control_time_step))
        self._sim_initialize(time_step, self._substeps_per_step, self.__headless)

    @abstractmethod
    def _sim_initialize(self, time_step: float, substeps_per_step: int, headless: bool):
        pass

    def add_ur10_robot(self, name: str, add_digit_sensors: bool = False, ur10_urdf_path: Optional[Path] = None,
                       rh_p12_rn_urdf_path: Optional[Path] = None) -> RobotType:
        """
        Adds the UR10 robot to the scene.
        :param name:                Name of the new robot
        :param add_digit_sensors:   Whether to add the Digit sensors.
        :param ur10_urdf_path:      The urdf to use for the ur10 robot arm; if None is passed, the urdf at
                                    robot_gym/res/ur10.urdf is used
        :param rh_p12_rn_urdf_path: The urdf to use for the rh_p12_rn gripper; if None is passed, the urdf at
                                    robot_gym/res/rh_p12_rn.urdf is used
        :return: The robot object.
        """
        robot = self._add_ur10_robot(name, add_digit_sensors, ur10_urdf_path, rh_p12_rn_urdf_path)
        self._register_robot(robot)
        return robot

    @abstractmethod
    def _add_ur10_robot(self, name: str, add_digit_sensors: bool, ur10_urdf_path: Optional[Path] = None,
                        rh_p12_rn_urdf_path: Optional[Path] = None) -> RobotType:
        pass

    @abstractmethod
    def add_simple_object(self, visual_shape: Optional[VisualShapeType] = None,
                          collision_shape: Optional[CollisionShapeType] = None,
                          mass: float = 0.0, friction: float = 0.5, restitution: float = 0.0) -> ObjectType:
        """
        Creates an simulation object from a visual shape and a collision shape.
        :param visual_shape:    Visual shape of this object. If this parameter is left None, the collision shape is used
                                as visualization.
        :param collision_shape: Collision shape of this object. If this parameter is left None, the resulting object
                                will not physically interact with the environment.
        :param mass:            Mass of the object. Pass 0 to make object stationary. If collision_shape is None, this
                                parameter will be ignored.
        :param friction:        Friction coefficient of this object.
        :param restitution:     Restitution coefficient of this object.
        :return: The object.
        """
        pass

    def create_visual_shape(
            self, shape_type: ShapeTypes,
            mesh_data: Optional[Union[Sequence[Trimesh], Sequence[str]]] = None,
            mesh_scale: float = 1.0,
            box_extents: Optional[Sequence[Sequence[float]]] = None,
            sphere_radii: Optional[Sequence[float]] = None,
            rgba_colors: Union[Sequence[Sequence[float]], Sequence[float]] = (1.0, 0.0, 0.0, 1.0)) -> VisualShapeType:
        """
        Creates a visual shape from a sequence of meshes. Note that shapes are only templates that do not appear in the
        scene immediately but can me used to create objects with the "add_simple_object" function.
        :param shape_type:      Type of this shape.
        :param mesh_data:       Meshes this shape consists of. Can either be a list of Trimesh instances or a list of
                                paths to individual *.obj files. Ignored if shape_type != ShapeTypes.MESH.
        :param mesh_scale:      Factor to scale mesh coordinates with.
        :param box_extents:     Nx3 list of x, y, z extents of the box. Ignored if shape_type != ShapeTypes.BOX.
        :param sphere_radii:    List of radii of the sphere. Ignored if shape_type != ShapeTypes.SPHERE.
        :param rgba_colors:     RGBA colors for each of the meshes. If a single 4D vector is given, all shapes are
                                colored in the same color
        :return: Visual shape that can be used in "add_simple_object".
        """
        if shape_type == ShapeTypes.MESH:
            assert mesh_data is not None
            count = len(mesh_data)
        elif shape_type == ShapeTypes.BOX:
            assert box_extents is not None
            count = len(box_extents)
            box_extents = np.array(box_extents)
            assert box_extents.shape == (count, 3)
        else:
            assert sphere_radii is not None
            count = len(sphere_radii)
            sphere_radii = np.array(sphere_radii)
        rgba_colors = np.array(rgba_colors)
        if len(rgba_colors.shape) == 1:
            rgba_colors = np.tile(rgba_colors, reps=(count, 1))
        assert rgba_colors.shape == (count, 4)
        return self._create_visual_shape(shape_type, rgba_colors, mesh_data, mesh_scale, box_extents, sphere_radii)

    @abstractmethod
    def _create_visual_shape(
            self, shape_type: ShapeTypes,
            rgba_colors: Sequence[np.ndarray],
            mesh_data: Optional[Union[Sequence[Trimesh], Sequence[str]]] = None,
            mesh_scale: float = 1.0,
            extents: Optional[np.ndarray] = None,
            radii: Optional[np.ndarray] = None) -> VisualShapeType:
        pass

    def create_collision_shape(
            self, shape_type: ShapeTypes,
            mesh_data: Optional[Union[Sequence[Trimesh], Sequence[str]]] = None,
            mesh_scale: float = 1.0,
            use_mesh_bounding_boxes: bool = False,
            box_extents: Optional[Sequence[Sequence[float]]] = None,
            sphere_radii: Optional[Sequence[float]] = None) -> CollisionShapeType:
        """
        Creates a visual shape from a sequence of meshes. Note that shapes are only templates that do not appear in the
        scene immediately but can me used to create objects with the "add_simple_object" function.
        :param shape_type:              Type of this shape.
        :param mesh_data:               Meshes this shape consists of. Can either be a list of Trimesh instances or a
                                        list of paths to individual *.obj files. Ignored if
                                        shape_type != ShapeTypes.MESH.
        :param mesh_scale:              Factor to scale mesh coordinates with.
        :param use_mesh_bounding_boxes: Whether to use the bounding boxes of the given meshes as collision shape instead
                                        of the actual meshes. If the meshes are rectangular this can cause a significant
                                        increase in performance.
        :param box_extents:             Nx3 list of x, y, z extents of the box. Ignored if shape_type != ShapeTypes.BOX.
        :param sphere_radii:            List of radii of the sphere. Ignored if shape_type != ShapeTypes.SPHERE.
        :return: Visual shape that can be used in "add_simple_object".
        """
        if shape_type == ShapeTypes.MESH:
            assert mesh_data is not None
        elif shape_type == ShapeTypes.BOX:
            assert box_extents is not None
            box_extents = np.array(box_extents)
            assert box_extents.shape[1:] == (3,)
        else:
            assert sphere_radii is not None
            sphere_radii = np.array(sphere_radii)
        return self._create_collision_shape(
            shape_type, mesh_data, mesh_scale, use_mesh_bounding_boxes, box_extents, sphere_radii)

    @abstractmethod
    def _create_collision_shape(
            self, shape_type: ShapeTypes,
            mesh_data: Optional[Union[Sequence[Trimesh], Sequence[str]]] = None,
            mesh_scale: float = 1.0,
            use_mesh_bounding_boxes: bool = False,
            extents: Optional[np.ndarray] = None,
            radii: Optional[np.ndarray] = None) -> CollisionShapeType:
        pass

    @ensure_initialized
    @abstractmethod
    def remove_object(self, object: ObjectType):
        pass

    @ensure_initialized
    def set_reset_checkpoint(self):
        """
        Sets the checkpoint the simulation is returning to when reset_scene() is called to the current state.
        :return:
        """
        self.__post_reset_robot_names = []
        self._set_reset_checkpoint()

    @abstractmethod
    def _set_reset_checkpoint(self):
        pass

    @ensure_initialized
    def start_reset(self):
        for name in self.__post_reset_robot_names:
            self._unregister_robot(name)
        self._reset_scene()

    @abstractmethod
    def _reset_scene(self):
        pass

    @ensure_initialized
    @abstractmethod
    def shutdown(self):
        pass

    @ensure_initialized
    def step(self):
        self._step()

    @abstractmethod
    def _step(self):
        pass

    @property
    def substeps_per_step(self) -> int:
        return self._substeps_per_step
