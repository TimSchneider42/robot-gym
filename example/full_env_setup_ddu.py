from rhp12rn import RHP12RNAConnector

from robot_gym.environment import Robot
from robot_gym.environment.real import RHP12RNGripper, UR10RobotArm, RealEnvironment, Optitrack

if __name__ == "__main__":
    gripper = RHP12RNGripper(RHP12RNAConnector("/dev/ttyUSB0", baud_rate=2000000, dynamixel_id=1))
    arm = UR10RobotArm("192.168.1.101", ur_cap_port=50002, optitrack_tcp_name="gripper")
    robot = Robot(arm, gripper, "ur10")
    optitrack = Optitrack(server_ip_address="192.168.1.35", local_ip_address="192.168.1.114", use_multicast=False)

    environment = RealEnvironment(optitrack, [robot])
    environment.initialize(0.2)

    # Do some stuff

    environment.shutdown()
