import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robot_gym.environment.real import UR10RobotArm
from robot_gym.util import Transformation


def test_config(ur10: UR10RobotArm, box_transformation: Transformation, box_extents: np.ndarray,
                rotation_center: Rotation):
    input("Robot will now move to box corners. Press enter to continue...")
    ur10.initialize()

    for x, y, z in [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]:
        corner_pos = box_extents / 2 * np.array([x, y, z])
        corner_pos_world_frame = box_transformation.transform(corner_pos)
        ur10.move_to_pose(Transformation(corner_pos_world_frame, rotation_center), velocity=0.05)
        time.sleep(1.0)

    ur10.shutdown()


def test(args):
    with Path(args.input_filename).open() as f:
        param_dict = json.load(f)
    ur10 = UR10RobotArm(args.robot_hostname, ur_cap_port=args.ur_cap_port)
    test_config(ur10, Transformation.from_dict(param_dict["box_transformation"]), np.array(param_dict["box_extents"]),
                Rotation.from_quat(param_dict["rotation_center"]))


def define_new(args):
    ur10 = UR10RobotArm(args.robot_hostname, ur_cap_port=args.ur_cap_port)
    ur10.initialize()

    input("Move to center rotation and press enter...")
    ur10.finish_observing()
    rotation_center = ur10.tcp_pose_robot_frame.rotation

    input("Move to first corner and press enter...")
    ur10.finish_observing()
    first_corner = ur10.tcp_pose_robot_frame.translation

    input("Move to an adjacent corner and press enter...")
    ur10.finish_observing()
    second_corner = ur10.tcp_pose_robot_frame.translation

    input(
        "Move to an edge parallel to the edge defined by the previous two points and press enter...")
    ur10.finish_observing()
    edge_pos = ur10.tcp_pose_robot_frame.translation

    input(
        "Move somewhere inside the surface opposite to the surface the previous three points are in and press enter...")
    ur10.finish_observing()
    surface_pos = ur10.tcp_pose_robot_frame.translation

    v_1 = (edge_pos - first_corner)
    v_2 = (second_corner - first_corner)
    v_2_norm = v_2 / np.linalg.norm(v_2)
    projection_dist = v_1.dot(v_2_norm)
    projection_pos = projection_dist * v_2_norm

    base_vec_1 = v_2_norm  # x
    base_vec_2_unnorm = v_1 - projection_pos
    base_vec_2 = base_vec_2_unnorm / np.linalg.norm(base_vec_2_unnorm)  # y
    base_vec_3_unnorm = np.cross(base_vec_1, base_vec_2_unnorm)
    base_vec_3 = base_vec_3_unnorm / np.linalg.norm(base_vec_3_unnorm)
    assert np.abs(base_vec_1.dot(base_vec_2)) < 1e-7
    assert np.abs(base_vec_2.dot(base_vec_3)) < 1e-7
    assert np.abs(base_vec_1.dot(base_vec_3)) < 1e-7

    box_corner_trans = Transformation(
        first_corner, Rotation.from_matrix(np.stack([base_vec_1, base_vec_2, base_vec_3], axis=1)))
    box_extents = np.array(
        [np.linalg.norm(v_2), np.linalg.norm(base_vec_2_unnorm), (surface_pos - first_corner).dot(base_vec_3)])
    box_center_trans_corner_frame = Transformation(box_extents / 2)
    box_extents_abs = np.abs(box_extents)
    box_center_trans_robot_frame = box_corner_trans.transform(box_center_trans_corner_frame)

    output_dict = {
        "box_transformation": box_center_trans_robot_frame.to_dict(),
        "box_extents": box_extents_abs.tolist(),
        "rotation_center": rotation_center.as_quat().tolist()
    }

    with Path(args.output_filename).open("w") as f:
        json.dump(output_dict, f)

    ur10.shutdown()
    test_config(ur10, box_center_trans_robot_frame, box_extents_abs, rotation_center)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Define parameters for the hand-eye calibration.")

    parser.add_argument("robot_hostname", type=str, help="Hostname of the robot.")
    parser.add_argument("-p", "--ur-cap-port", type=int, default=50002, help="Port of URCap on the robot.")

    subparsers = parser.add_subparsers()

    parser_new = subparsers.add_parser("new")
    parser_new.add_argument("output_filename", type=str, help="Where to store the hand-eye calibration definition.")
    parser_new.set_defaults(func=define_new)

    parser_test = subparsers.add_parser("test")
    parser_test.add_argument("input_filename", type=str, help="Hand-eye calibration file to load.")
    parser_test.set_defaults(func=test)

    args = parser.parse_args()
    args.func(args)
