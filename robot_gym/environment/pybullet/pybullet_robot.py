from .pybullet_robot_arm import PyBulletRobotArm
from .pybullet_object import PybulletObject
from .pybullet_robot_component import PybulletRobotComponent
from robot_gym.environment.simulation import SimulationRobot


class PyBulletRobot(SimulationRobot[PyBulletRobotArm, PybulletRobotComponent], PybulletObject):
    def __init__(self, arm: PyBulletRobotArm, gripper: PybulletRobotComponent, name: str):
        SimulationRobot.__init__(self, arm, gripper, name)
        PybulletObject.__init__(self, arm.wrapped_body)
