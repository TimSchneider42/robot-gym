import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence, List, Union

import pybullet_data
import trimesh

import numpy as np
from scipy.spatial.transform import Rotation
from trimesh import Trimesh

import pybullet
from .pybullet_robot_arm import PyBulletRobotArm

from pyboolet import PhysicsClient, VisualShapeArray, CollisionShapeArray
from pyboolet.multibody import URDFBody, SimpleBody, Multibody
from .pybullet_gripper_joint import PybulletGripperJoint
from .pybullet_joint import PybulletJoint
from .pybullet_object import PybulletObject
from .pybullet_robot import PyBulletRobot
from .pybullet_robot_component import PybulletRobotComponent
from .urdf_editor import UrdfEditor
from robot_gym.util import Event
from transformation import Transformation
from robot_gym.environment.simulation import SimulationEnvironment, ShapeTypes


class PybulletEnvironment(SimulationEnvironment[PybulletObject, PyBulletRobot, VisualShapeArray, CollisionShapeArray]):
    def __init__(self, real_time_factor: Optional[float] = None, simulator_step_s: float = 0.005,
                 headless: bool = True):
        self.__pc: Optional[PhysicsClient] = None
        self.__bg_pc: Optional[PhysicsClient] = None
        self.__bg_arm: Optional[URDFBody] = None
        self.__arm_loaded = False
        self.__reset_checkpoint: Optional[int] = None
        self.__previous_step = None
        self.__real_time_factor = real_time_factor
        self.__virtual_substep_mode = False
        self.__on_step_event = Event()
        self.__pb_objects_added_after_reset: Optional[List[Multibody]] = None
        self.__temp_dir = Path(tempfile.mkdtemp())
        self.__full_arm_path = self.__temp_dir / "ur10_rh_p12_rn.urdf"
        self.ground_plane: Optional[URDFBody] = None
        super(PybulletEnvironment, self).__init__("pybullet", simulator_step_s=simulator_step_s, headless=headless)

    def _sim_initialize(self, timestep: float, substeps_per_step: int, headless: bool):
        super(PybulletEnvironment, self)._sim_initialize(timestep, substeps_per_step, headless)
        self.__on_step_event = Event()
        self.__pb_objects_added_after_reset = []

        self.__pc = PhysicsClient()
        if headless:
            self.__pc.connect_direct()
        else:
            self.__pc.connect_gui()
        self.__pc.gravity = np.array([0, 0, -9.81])

        self.__pc.set_additional_search_path(pybullet_data.getDataPath())
        with self.__pc.as_default():
            self.ground_plane = URDFBody("plane.urdf", use_fixed_base=True)

        # Settings the virtual substep mode. Note that this sets the time step as well
        # self.virtual_substep_mode = self.__real_time_factor is not None
        self.virtual_substep_mode = True  # Needs to be set to enforce the acceleration limits

    def _add_ur10_robot(self, name: str, add_digit_sensors: bool = False, ur10_urdf_path: Optional[Path] = None,
                        rh_p12_rn_urdf_path: Optional[Path] = None) -> PyBulletRobot:
        if not self.__arm_loaded:
            tmp_pc = PhysicsClient()
            tmp_pc.connect_direct()

            model_path = Path(__file__).parent.parent.parent / "res"

            with tmp_pc.as_default():
                if ur10_urdf_path is not None:
                    arm = URDFBody(str(ur10_urdf_path))
                else:
                    arm = URDFBody(str(model_path / "ur10.urdf"))
                if rh_p12_rn_urdf_path is not None:
                    gripper = URDFBody(str(rh_p12_rn_urdf_path))
                else:
                    gripper = URDFBody(str(model_path / "rh_p12_rn.urdf"))

            ed_arm = UrdfEditor()
            ed_arm.initializeFromBulletBody(arm.unique_id, tmp_pc.physics_client_id)

            ed_gripper = UrdfEditor()
            ed_gripper.initializeFromBulletBody(gripper.unique_id, tmp_pc.physics_client_id)

            if self.__bg_pc is None:
                self.__bg_pc = PhysicsClient()
                self.__bg_pc.connect_direct()
                with self.__bg_pc.as_default():
                    self.__bg_arm = URDFBody(str(model_path / "ur10.urdf"))

            # TODO: why is the +1 neccessary?
            new_joint = ed_arm.joinUrdf(ed_gripper, parentLinkIndex=arm.links["wrist_3_link"].link_index + 1,
                                        jointPivotRPYInParent=[-np.pi / 2, 0, 0],
                                        jointPivotRPYInChild=[0, 0, 0],
                                        jointPivotXYZInParent=[0, 8.2515e-02, 0],
                                        parentPhysicsClientId=tmp_pc.physics_client_id,
                                        childPhysicsClientId=tmp_pc.physics_client_id)
            new_joint.joint_type = pybullet.JOINT_FIXED
            new_joint.joint_name = "gripper_wrist_joint"

            ed_arm.saveUrdf(self.__full_arm_path)
            tmp_pc.disconnect()

            self.__arm_loaded = True

        with self.__pc.as_default():
            robot = URDFBody(
                self.__full_arm_path, use_fixed_base=True,
                flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL |
                      pybullet.URDF_USE_SELF_COLLISION |
                      pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)

        # Sticky fingers
        for l in ["rh_p12_rn_r2", "rh_p12_rn_r2"]:
            if l in robot.links:
                robot.links[l].change_dynamics(lateral_friction=1.0, spinning_friction=1.0)

        arm = PyBulletRobotArm(robot, tuple(PybulletJoint(j) for j in robot.revolute_joints[:6]),
                               self.__bg_arm, self.time_step, boundary_link=robot.joints["gripper_wrist_joint"].child)
        home_position = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
        arm.move_to_joint_positions(home_position)
        arm.set_joint_target_positions(home_position)

        gripper = PybulletRobotComponent(
            robot, (PybulletGripperJoint([PybulletJoint(j) for j in robot.revolute_joints[6:]]),),
            robot.joints["gripper_wrist_joint"].child)
        wrapped_robot = PyBulletRobot(arm, gripper, name)
        self.__pb_objects_added_after_reset.append(robot)
        return wrapped_robot

    def add_simple_object(self, visual_shape: Optional[VisualShapeArray] = None,
                          collision_shape: Optional[CollisionShapeArray] = None, mass: float = 0.0,
                          friction: float = 0.5, restitution: float = 1.0) -> PybulletObject:
        with self.__pc.as_default():
            pb_object = SimpleBody(collision_shape, visual_shape, mass=mass)
            pb_object.change_dynamics(lateral_friction=friction, spinning_friction=friction, restitution=restitution)
        self.__pb_objects_added_after_reset.append(pb_object)
        return PybulletObject(pb_object)

    def _create_shape(
            self, visual: bool, shape_type: ShapeTypes,
            mesh_data: Optional[Union[Sequence[Trimesh], Sequence[str]]] = None, mesh_scale: float = 1.0,
            use_mesh_bounding_boxes: bool = False, extents: Optional[np.ndarray] = None,
            radii: Optional[np.ndarray] = None, **kwargs) \
            -> Union[VisualShapeArray, CollisionShapeArray]:
        if shape_type == ShapeTypes.BOX:
            kwargs.update({
                "shape_type": [pybullet.GEOM_BOX] * extents.shape[0],
                "halfExtents": extents / 2
            })
        elif shape_type == ShapeTypes.SPHERE:
            kwargs.update({
                "shape_type": [pybullet.GEOM_SPHERE] * radii.shape[0],
                "radii": radii
            })
        else:
            if not use_mesh_bounding_boxes:
                if isinstance(mesh_data[0], str):
                    assert all(isinstance(m, str) for m in mesh_data)
                    mesh_filenames = mesh_data
                else:
                    assert all(isinstance(m, Trimesh) for m in mesh_data)
                    mesh_dir = None
                    while mesh_dir is None or mesh_dir.exists():
                        mesh_dir = self.__temp_dir / "".join(random.choices("0123456789", k=16))
                    mesh_dir.mkdir()
                    mesh_filenames = []
                    for j, sub_mesh in enumerate(mesh_data):
                        filename = mesh_dir / "{}.obj".format(j)
                        mesh_filenames.append(str(filename))
                        with filename.open("w") as f:
                            sub_mesh.export(f, file_type="obj")
                kwargs.update({
                    "shape_type": [pybullet.GEOM_MESH] * len(mesh_filenames),
                    "fileNames": mesh_filenames,
                    "meshScales": [mesh_scale] * len(mesh_filenames)
                })
            else:
                if isinstance(mesh_data[0], str):
                    assert all(isinstance(m, str) for m in mesh_data)
                    meshes = [trimesh.load(m).apply_scale(mesh_scale) for m in mesh_data]
                else:
                    assert all(isinstance(m, Trimesh) for m in mesh_data)
                    meshes = mesh_data
                min_vertex_positions = np.array([np.min(s.vertices, axis=0) for s in meshes])
                max_vertex_positions = np.array([np.max(s.vertices, axis=0) for s in meshes])
                dims = max_vertex_positions - min_vertex_positions
                positions = min_vertex_positions + dims / 2
                kwargs.update({
                    "shape_type": [pybullet.GEOM_BOX] * len(meshes),
                    "halfExtents": dims / 2,
                    "collisionFramePositions": positions
                })
        with self.__pc.as_default():
            if visual:
                return VisualShapeArray(**kwargs)
            else:
                return CollisionShapeArray(**kwargs)

    def _create_visual_shape(self, shape_type: ShapeTypes, rgba_colors: Sequence[np.ndarray],
                             mesh_data: Optional[Union[Sequence[Trimesh], Sequence[str]]] = None,
                             mesh_scale: float = 1.0,
                             extents: Optional[np.ndarray] = None,
                             radii: Optional[np.ndarray] = None) -> VisualShapeArray:
        # noinspection PyTypeChecker
        return self._create_shape(visual=True, shape_type=shape_type, mesh_data=mesh_data, mesh_scale=mesh_scale,
                                  use_mesh_bounding_boxes=False, extents=extents, radii=radii, rgbaColors=rgba_colors)

    def _create_collision_shape(self, shape_type: ShapeTypes,
                                mesh_data: Optional[Union[Sequence[Trimesh], Sequence[str]]] = None,
                                mesh_scale: float = 1.0,
                                use_mesh_bounding_boxes: bool = False,
                                extents: Optional[np.ndarray] = None,
                                radii: Optional[np.ndarray] = None) -> VisualShapeArray:
        # noinspection PyTypeChecker
        return self._create_shape(visual=False, shape_type=shape_type, mesh_data=mesh_data, mesh_scale=mesh_scale,
                                  use_mesh_bounding_boxes=use_mesh_bounding_boxes, extents=extents, radii=radii)

    def remove_object(self, obj: PybulletObject):
        assert obj.wrapped_root_link is None or obj.boundary_link is None, "Cannot remove parts of an object."
        assert obj.wrapped_body in self.__pb_objects_added_after_reset, \
            "Cannot remove objects that have been added before the reset checkpoint was created."
        obj.wrapped_body.remove()
        self.__pb_objects_added_after_reset.remove(obj.wrapped_body)

    def _set_reset_checkpoint(self):
        self.__pb_objects_added_after_reset = []
        self.__reset_checkpoint = self.__pc.call(pybullet.saveState)

    def _reset_scene(self):
        for o in self.__pb_objects_added_after_reset:
            o.remove()
        self.__pb_objects_added_after_reset.clear()
        self.__pc.call(pybullet.restoreState, stateId=self.__reset_checkpoint)
        self.__previous_step = None

    def shutdown(self):
        if self.__pc.is_connected:
            self.__pc.disconnect()
        if self.__bg_pc.is_connected:
            self.__bg_pc.disconnect()

    def _step(self):
        if not self.__virtual_substep_mode:
            now = time.time()
            if self.__real_time_factor is not None and self.__previous_step is not None:
                next_step = self.__previous_step + self.time_step / self.__real_time_factor
                time.sleep(max(0, next_step - now))
                self.__previous_step = next_step
            else:
                self.__previous_step = now
            for r in self.robots.values():
                r.arm.update()
            self.__pc.step_simulation()
            self.__on_step_event(self)
        else:
            if self.__real_time_factor is not None:
                step_time = self.time_step / self.__real_time_factor / self.substeps_per_step
            else:
                step_time = 0
            for _ in range(self.substeps_per_step):
                now = time.time()
                if self.__real_time_factor is not None and self.__previous_step is not None:
                    next_step = self.__previous_step + step_time
                    time.sleep(max(0, next_step - now))
                    self.__previous_step = next_step
                else:
                    self.__previous_step = now
                for r in self.robots.values():
                    r.arm.update()
                self.__pc.step_simulation()
                self.__on_step_event(self)

    @property
    def physics_client(self) -> PhysicsClient:
        return self.__pc

    @property
    def virtual_substep_mode(self) -> bool:
        return self.__virtual_substep_mode

    @virtual_substep_mode.setter
    def virtual_substep_mode(self, value: bool):
        self.reconfigure_time_step(self.time_step, self.substeps_per_step, value)

    @property
    def on_step_event(self) -> Event:
        return self.__on_step_event

    def reconfigure_time_step(self, new_time_step: float, new_substeps_per_step: int, new_virtual_step_mode: bool):
        if new_virtual_step_mode:
            self.__pc.call(pybullet.setPhysicsEngineParameter, numSubSteps=1)
            self.__pc.time_step = new_time_step / new_substeps_per_step
        else:
            self.__pc.call(pybullet.setPhysicsEngineParameter, numSubSteps=new_substeps_per_step)
            self.__pc.time_step = new_time_step
        self.__virtual_substep_mode = new_virtual_step_mode
        self._simulator_step_s = new_time_step
        self._substeps_per_step = new_substeps_per_step

    def __del__(self):
        shutil.rmtree(self.__temp_dir)
