from robot_gym.environment.generic import RobotComponent
from .joint_velocity_sensor import JointVelocitySensor


class GripperJointVelocitySensor(JointVelocitySensor):
    def __init__(self, robot_name: str = "ur10"):
        super(GripperJointVelocitySensor, self).__init__(name_prefix="gripper")
        self.__robot_name = robot_name

    def _get_observed_robot_component(self) -> RobotComponent:
        return self.task.environment.robots[self.__robot_name].gripper
