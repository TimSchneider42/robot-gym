from enum import Enum


class JointMode(Enum):
    UNKNOWN = -1
    TORQUE_CONTROL = 0
    VELOCITY_CONTROL = 1
    POSITION_CONTROL = 2
