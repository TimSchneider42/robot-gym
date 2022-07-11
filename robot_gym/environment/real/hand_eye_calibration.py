from typing import Sequence, Tuple, NamedTuple, Union, List, Dict, Any, Literal, Optional

import numpy as np
import torch
from scipy.optimize import minimize, NonlinearConstraint
from scipy.spatial.transform import Rotation

from transformation import Transformation
from robot_gym import logger


class HandEyeCalibrationResult(NamedTuple):
    robot_pose_world_frame: Transformation
    marker_pose_tcp_frame: Transformation
    mean_translational_error: float
    mean_rotational_error: float
    recorded_tcp_poses_robot_frame: Tuple[Transformation, ...]
    recorded_marker_poses_world_frame: Tuple[Transformation, ...]

    def _to_json(self, element: Union[float, Tuple, Transformation]):
        if isinstance(element, float):
            return element
        elif isinstance(element, tuple):
            return [self._to_json(e) for e in element]
        else:
            return element.to_dict()

    @classmethod
    def _parse_element(cls, name: str, value: Union[float, List, Dict[str, Any]]):
        field_type = cls.__annotations__[name]
        if field_type == float:
            return float(value)
        elif field_type == Transformation:
            return Transformation.from_dict(value)
        else:
            return tuple(Transformation.from_dict(e) for e in value)

    def to_dict(self):
        return {k: self._to_json(v) for k, v in self._asdict().items()}

    @classmethod
    def from_dict(cls, value_dict: Dict[str, Any]):
        return cls(**{k: cls._parse_element(k, v) for k, v in value_dict.items()})


def generate_random_pose(generator, trange, rrange):
    translation_direction = generator.random(3)
    translation_length = generator.random(1) * trange
    translation = translation_direction * translation_length
    rotation_direction = generator.random(3)
    rotation_amount = generator.random(1) * rrange
    rotation_rotvec = rotation_direction * rotation_amount
    pose = Transformation.from_pos_rotvec(translation, rotation_rotvec)
    return pose


def quad_to_matrix_right_mult(quaternion: np.ndarray):
    """
    Converts a quaternion q into a matrix A(q), such that for another quaternion q': q*q'=A(q)q'.
    :param quaternion:
    :return:
    """
    q = quaternion
    return np.stack([
        [q[..., 3], -q[..., 2], q[..., 1], q[..., 0]],
        [q[..., 2], q[..., 3], -q[..., 0], q[..., 1]],
        [-q[..., 1], q[..., 0], q[..., 3], q[..., 2]],
        [-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]]]).transpose((2, 0, 1))


def quad_to_matrix_left_mult(quaternion: np.ndarray):
    """
    Converts a quaternion q into a matrix A(q), such that for another quaternion q': q'*q=A(q)q'.
    :param quaternion:
    :return:
    """
    q = quaternion
    return np.stack([
        [q[..., 3], q[..., 2], -q[..., 1], q[..., 0]],
        [-q[..., 2], q[..., 3], q[..., 0], q[..., 1]],
        [q[..., 1], -q[..., 0], q[..., 3], q[..., 2]],
        [-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]]]).transpose((2, 0, 1))


def compute_pose_error(pose: Transformation, target_pose: Transformation) -> Tuple[float, float]:
    """
    Computes the position error in m and the rotation error in rad from pose to target_pose.
    :param pose:
    :param target_pose:
    :return:
    """
    position_error = np.linalg.norm(target_pose.translation - pose.translation)
    rot_error_angle = np.arccos(2 * np.minimum(pose.quaternion.dot(target_pose.quaternion) ** 2, 1) - 1)
    return position_error, rot_error_angle


def np_cache(fun):
    fun.cache_val = None
    fun.cache_inp = None

    def wrapper(x: np.ndarray):
        if fun.cache_inp is None or not np.array_equal(x, fun.cache_inp):
            fun.cache_val = fun(x)
            fun.cache_inp = x
        return fun.cache_val

    return wrapper


def _generate_quad_error_matrices(tcp_pose_samples_robot_frame: Sequence[Transformation],
                                  marker_pose_samples_world_frame: Sequence[Transformation]):
    tcp_quaternions_robot_frame = np.stack([pose.quaternion for pose in tcp_pose_samples_robot_frame], axis=0)

    marker_world_frame_quats = np.stack([pose.quaternion for pose in marker_pose_samples_world_frame], axis=0)

    tcp_robot_frame_qml = quad_to_matrix_left_mult(tcp_quaternions_robot_frame)
    marker_world_frame_qmr = quad_to_matrix_right_mult(marker_world_frame_quats)

    # The quaternion error can be expressed as 1 - (q_tm^T A q_rw2)^2 where A = R(q_mw)^T L(q_tr)
    return marker_world_frame_qmr.transpose((0, 2, 1)) @ tcp_robot_frame_qml


def _estimate_hand_eye_calibration_rotation_torch(
        tcp_pose_samples_robot_frame: Sequence[Transformation],
        marker_pose_samples_world_frame: Sequence[Transformation],
        fixed_marker_pose_tcp_frame: Optional[Transformation] = None) -> Tuple[Rotation, Rotation]:
    error_matrices = _generate_quad_error_matrices(tcp_pose_samples_robot_frame, marker_pose_samples_world_frame)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    error_matrices_torch = torch.from_numpy(error_matrices).to(dev)
    robot_quat_world_frame = torch.autograd.Variable(torch.randn(4, dtype=torch.double, device=dev), requires_grad=True)
    if fixed_marker_pose_tcp_frame:
        tcp_quat_marker_frame = torch.from_numpy(fixed_marker_pose_tcp_frame.inv.quaternion).double().to(dev)
        variables = [robot_quat_world_frame]
    else:
        tcp_quat_marker_frame = torch.autograd.Variable(
            torch.randn(4, dtype=torch.double, device=dev), requires_grad=True)
        variables = [robot_quat_world_frame, tcp_quat_marker_frame]

    optimizer = torch.optim.Adam(variables)

    for i in range(10000):
        optimizer.zero_grad()

        q_rw = robot_quat_world_frame / torch.norm(robot_quat_world_frame)
        q_tm = tcp_quat_marker_frame / torch.norm(tcp_quat_marker_frame)

        error: torch.Tensor = 1 - (q_tm[None, None] @ error_matrices_torch @ q_rw[None, :, None]) ** 2
        error_sum = error.sum()
        error_sum.backward()

        optimizer.step()

    with torch.no_grad():
        q_rw = robot_quat_world_frame / torch.norm(robot_quat_world_frame)
        q_tm = tcp_quat_marker_frame / torch.norm(tcp_quat_marker_frame)

        robot_rotation_world_frame = Rotation.from_quat(q_rw.cpu().numpy())
        tcp_rotation_marker_frame = Rotation.from_quat(q_tm.cpu().numpy())

    return robot_rotation_world_frame, tcp_rotation_marker_frame


def _estimate_hand_eye_calibration_rotation_scipy(tcp_pose_samples_robot_frame: Sequence[Transformation],
                                                  marker_pose_samples_world_frame: Sequence[Transformation]) \
        -> Tuple[Rotation, Rotation]:
    error_matrices = _generate_quad_error_matrices(tcp_pose_samples_robot_frame, marker_pose_samples_world_frame)

    def make_quat_len_const(quat_slice: slice):
        def fun(q, s=quat_slice):
            return q[s].dot(q[s]) - 1

        def jac(q, s=quat_slice):
            jac = np.zeros_like(q)
            jac[s] = 2 * q[s]
            return jac

        def hess(q, v, s=quat_slice):
            hess = np.zeros((q.shape[0],) * 2)
            hess[s, s] = 2 * np.eye(4)
            return hess

        return NonlinearConstraint(fun, 0, 0, jac=jac, hess=hess)

    @np_cache
    def rot_error(quats: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        robot_world_frame, tcp_marker_frame = np.split(quats, 2)
        q_tm = tcp_marker_frame[None, :, None]
        q_tm_t = tcp_marker_frame[None, None]
        q_rw = robot_world_frame[None, :, None]
        q_rw_t = robot_world_frame[None, None]
        a = error_matrices
        a_t = error_matrices.transpose((0, 2, 1))

        quat_dot = q_tm_t @ a @ q_rw

        jac_tm = (a @ q_rw * quat_dot).sum(0).reshape(-1)
        jac_rw = (a_t @ q_tm * quat_dot).sum(0).reshape(-1)
        jac = -2 * np.concatenate([jac_rw, jac_tm])

        hess_tmtm = (a @ q_rw @ q_rw_t @ a_t).sum(0)
        hess_rwrw = (a_t @ q_tm @ q_tm_t @ a).sum(0)
        hess_tmrw = (a_t @ q_tm @ q_rw_t @ a_t + quat_dot * a_t).sum(0)

        hess = -2 * np.block([[hess_rwrw, hess_tmrw],
                              [hess_tmrw.T, hess_tmtm]])

        return (1 - quat_dot ** 2).sum(), jac, hess

    # I checked the gradients and they are correct, but still the performance becomes much worse when using the hessian.
    # So I will leave it out for now.
    x0 = np.random.rand(8) * 2 - 1
    x0[:4] /= np.linalg.norm(x0[:4])
    x0[4:] /= np.linalg.norm(x0[4:])
    res = minimize(lambda x: rot_error(x)[:2], x0, method="trust-constr", jac=True,
                   # hess=lambda x: rot_error(x)[2],
                   constraints=[make_quat_len_const(slice(4)), make_quat_len_const(slice(4, 8))],
                   options={"maxiter": 25000})
    if not res.success:
        logger.warning("The optimizer failed to find the minimum during the rotation estimation.")

    robot_rotation_world_frame = Rotation.from_quat(res.x[:4] / np.linalg.norm(res.x[:4]))
    tcp_rotation_marker_frame = Rotation.from_quat(res.x[4:] / np.linalg.norm(res.x[4:]))

    return robot_rotation_world_frame, tcp_rotation_marker_frame


def estimate_hand_eye_calibration(
        tcp_pose_samples_robot_frame: Sequence[Transformation],
        marker_pose_samples_world_frame: Sequence[Transformation],
        fixed_marker_pose_tcp_frame: Optional[Transformation] = None,
        rotation_estimation_backend: Literal["torch", "scipy"] = "torch") -> HandEyeCalibrationResult:
    if rotation_estimation_backend == "torch":
        robot_rotation_world_frame, tcp_rotation_marker_frame = _estimate_hand_eye_calibration_rotation_torch(
            tcp_pose_samples_robot_frame, marker_pose_samples_world_frame,
            fixed_marker_pose_tcp_frame=fixed_marker_pose_tcp_frame)
    else:
        assert fixed_marker_pose_tcp_frame is None, "scipy backend does not support fixed marker pose."
        robot_rotation_world_frame, tcp_rotation_marker_frame = _estimate_hand_eye_calibration_rotation_scipy(
            tcp_pose_samples_robot_frame, marker_pose_samples_world_frame)

    rotated_tcp_pos = robot_rotation_world_frame.apply([pose.translation for pose in tcp_pose_samples_robot_frame])
    marker_trans_world_frame = np.stack([pose.translation for pose in marker_pose_samples_world_frame])
    marker_rot_world_frame = np.stack([pose.rotation.as_matrix() for pose in marker_pose_samples_world_frame], axis=0)

    if fixed_marker_pose_tcp_frame is None:
        # Determine translations
        # Formulate problem as min x: sum(i = 0 -> N) ||A x - b||^2

        A = np.concatenate(
            [marker_rot_world_frame, np.broadcast_to(-np.eye(3, 3)[None], (marker_rot_world_frame.shape[0], 3, 3))],
            axis=2)

        b = rotated_tcp_pos - marker_trans_world_frame

        a_trans = A.transpose((0, 2, 1))
        left_side = (a_trans @ A).sum(0)
        right_side = (a_trans @ b[:, :, None]).sum(0)

        x = np.linalg.solve(left_side, right_side)[:, 0]

        tcp_trans_marker_frame = x[:3]
        robot_trans_world_frame = x[3:]
    else:
        tcp_trans_marker_frame = fixed_marker_pose_tcp_frame.inv.translation

        # Optimal solution simply becomes average over robot translations calculated from each recorded pose
        robot_trans_world_frame_proposals = \
            marker_rot_world_frame @ tcp_trans_marker_frame + marker_trans_world_frame - rotated_tcp_pos
        robot_trans_world_frame = np.mean(robot_trans_world_frame_proposals, axis=0)

    robot_pose_world_frame = Transformation(robot_trans_world_frame, robot_rotation_world_frame)
    marker_pose_tcp_frame = Transformation(tcp_trans_marker_frame, tcp_rotation_marker_frame).inv

    errors = np.array([compute_pose_error(
        robot_pose_world_frame * tcp_pose_rf * marker_pose_tcp_frame, marker_pose_wf)
        for marker_pose_wf, tcp_pose_rf in zip(marker_pose_samples_world_frame, tcp_pose_samples_robot_frame)])

    mean_trans_error = np.mean(errors[:, 0]).item()
    mean_ang_error = np.mean(errors[:, 1]).item()

    return HandEyeCalibrationResult(robot_pose_world_frame, marker_pose_tcp_frame, mean_trans_error, mean_ang_error,
                                    tuple(tcp_pose_samples_robot_frame), tuple(marker_pose_samples_world_frame))
