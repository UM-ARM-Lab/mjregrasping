import numpy as np
from mujoco import mju_negQuat, mju_mulQuat, mju_quat2Mat, mju_mat2Quat
from transformations import quaternion_from_euler as _wxyz_quaternion_from_euler
from transformations import quaternion_from_matrix as _wxyz_quaternion_from_matrix

from tf.transformations import quaternion_from_euler as _xyzw_quaternion_from_euler
from tf.transformations import quaternion_from_matrix as _xyzw_quaternion_from_matrix


def xyzw_quat_from_euler(roll, pitch, yaw):
    return _xyzw_quaternion_from_euler(roll, pitch, yaw)


def wxyz_quat_from_euler(roll, pitch, yaw):
    return _wxyz_quaternion_from_euler(roll, pitch, yaw)


def xyzw_quat_from_matrix(matrix):
    return _xyzw_quaternion_from_matrix(matrix)


def wxyz_quat_from_matrix(matrix):
    return _wxyz_quaternion_from_matrix(matrix)


def np_xyzw_to_wxyz(quat):
    x = quat[..., 0]
    y = quat[..., 1]
    z = quat[..., 2]
    w = quat[..., 3]
    return np.stack([w, x, y, z], axis=-1)


def np_wxyz_to_xyzw(quat):
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]
    return np.stack([x, y, z, w], axis=-1)



def matrix_dist(m1, m2):
    rot1 = m1[:3, :3]
    rot2 = m2[:3, :3]
    rot_dist = np.arccos(((rot1 * rot2.T).trace() - 1) / 2)
    pos1 = m1[:3, 3]
    pos2 = m2[:3, 3]
    pos_dist = np.linalg.norm(pos1 - pos2)
    return rot_dist * 0.1 + pos_dist


def pos_mat_to_matrix(pos, mat):
    """

    Args:
        pos: [x, y, z]
        mat: flattened 3x3 rotation matrix

    Returns: [4x4] homogeneous transformation matrix

    """
    full_matrix = np.eye(4)
    full_matrix[:3, :3] = mat.reshape(3, 3)
    full_matrix[:3, 3] = pos
    return full_matrix


def divide_quat(qa, qb):
    neg_qb = np.zeros(4)
    q_diff = np.zeros(4)
    mju_negQuat(neg_qb, qb)
    mju_mulQuat(q_diff, qa, neg_qb)
    return q_diff


def quaternion_difference(child_quat, parent_quat):
    # expected usage: parent2child_quat = quaternion_difference(child_quat, parent_quat)
    parent_rot = np.zeros(9)
    mju_quat2Mat(parent_rot, parent_quat)
    parent_rot = parent_rot.reshape(3, 3)
    child_rot = np.zeros(9)
    mju_quat2Mat(child_rot, child_quat)
    child_rot = child_rot.reshape(3, 3)
    parent2child = parent_rot.T @ child_rot
    parent2child_quat = np.zeros(4)
    mju_mat2Quat(parent2child_quat, parent2child.flatten())
    return parent2child_quat
