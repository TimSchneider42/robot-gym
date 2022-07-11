import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robot_gym.environment.real import Optitrack, UR10RobotArm
from robot_gym.environment.real.hand_eye_calibration import estimate_hand_eye_calibration
from transformation import Transformation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hec_params_file", type=str, help="File containing the parameters for the hand-eye calibration")
    parser.add_argument("optitrack_remote_ip", type=str, help="IP of the Optitrack server.")
    parser.add_argument("optitrack_local_ip", type=str, help="Local IP to use for Optitrack.")
    parser.add_argument("robot_hostname", type=str, help="Hostname of the robot.")
    parser.add_argument("output_path", type=str, help="Where to store the calibration result.")
    parser.add_argument("-p", "--ur-cap-port", type=int, default=50002, help="Port of URCap on the robot.")
    parser.add_argument("-c", "--sample-count", type=int, default=100, help="Number of pose samples to draw.")
    parser.add_argument("-n", "--optitrack-tcp-name", type=str, default="tcp",
                        help="Name of the rigid body attached to the robot TCP.")
    parser.add_argument("-f", "--finetune", type=str, help="Old hand_eye_calibration_result to finetune.")
    parser.add_argument("-r", "--recalculate", type=str,
                        help="hand_eye_calibration_result containing the poses to use.")
    args = parser.parse_args()

    if args.finetune is not None:
        with Path(args.finetune).open() as f:
            hec_res = json.load(f)
        marker_pose_tcp_frame = Transformation.from_dict(hec_res["marker_pose_tcp_frame"])
    else:
        marker_pose_tcp_frame = None

    if args.recalculate is not None:
        with Path(args.recalculate).open() as f:
            hec_res = json.load(f)
        tcp_poses = [Transformation.from_dict(e) for e in hec_res["recorded_tcp_poses_robot_frame"]]
        marker_poses = [Transformation.from_dict(e) for e in hec_res["recorded_marker_poses_world_frame"]]
        hand_eye_calibration_result = estimate_hand_eye_calibration(tcp_poses, marker_poses, marker_pose_tcp_frame)
        mean_ang_err_deg = hand_eye_calibration_result.mean_rotational_error / np.pi * 180
        print("Mean linear pose error: {:0.2f}mm".format(
            hand_eye_calibration_result.mean_translational_error * 1000))
        print("Mean angular pose error: {:0.4f} deg".format(mean_ang_err_deg))
    else:
        with Path(args.hec_params_file).open() as f:
            param_dict = json.load(f)

        optitrack = Optitrack(args.optitrack_remote_ip, args.optitrack_local_ip, use_multicast=False,
                              world_transformation=Transformation.from_pos_euler(euler_angles=[np.pi / 2, 0, 0]))
        optitrack.initialize()
        box_extents = np.array(param_dict["box_extents"])
        box_pose = Transformation.from_dict(param_dict["box_transformation"])
        center_quat = Rotation.from_quat(param_dict["rotation_center"])

        ur10 = UR10RobotArm(args.robot_hostname, ur_cap_port=args.ur_cap_port,
                            optitrack_tcp_name=args.optitrack_tcp_name,
                            calibration_safe_box_extents=box_extents, calibration_safe_box_pose=box_pose,
                            calibration_safe_orientation_center=center_quat, calibration_velocity=0.3,
                            calibration_safe_rotation_angle=np.pi / 4)
        ur10.initialize()
        hand_eye_calibration_result = ur10.run_hand_eye_calibration(
            optitrack, args.sample_count, fixed_marker_pose_tcp_frame=marker_pose_tcp_frame)

        optitrack.shutdown()
        ur10.shutdown()
    with Path(args.output_path).open("w") as f:
        json.dump(hand_eye_calibration_result.to_dict(), f)
