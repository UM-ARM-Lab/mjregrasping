import numpy as np
from numpy.linalg import norm

from mjregrasping.geometry import squared_norm


def skeleton_field_dir(skeleton, r):
    """
    Computes the field direction at the input points, where the conductor is the skeleton of an obstacle.
    A skeleton is defined by a set of points in 3D, like a line-strip, and can represent only a genus-1 obstacle (donut)
    Assumes μ and I are 1.
    Based on this paper: https://www.roboticsproceedings.org/rss07/p02.pdf

    Variables in my code <--> math in the paper:

        s_prev = s_i^j
        s_next = s_i^j'
        p_prev = p
        p_next = p'

    Args:
        skeleton: [n, 3] the points that define the skeleton
        r: [b, 3] the points at which to compute the field.
    """
    if not np.all(skeleton[0] == skeleton[-1]):
        raise ValueError("Skeleton must be a closed loop! Add the first point to the end.")

    s_prev = skeleton[:-1][None]  # [1, n, 3]
    s_next = skeleton[1:][None]  # [1, n, 3]

    p_prev = s_prev - r[:, None]  # [b, n, 3]
    p_next = s_next - r[:, None]  # [b, n, 3]
    squared_segment_lens = squared_norm(s_next - s_prev, keepdims=True)
    d = np.cross((s_next - s_prev), np.cross(p_next, p_prev)) / squared_segment_lens  # [b, n, 3]

    # bs is a matrix [b, n,3] where each bs[i, j] corresponds to a line segment in the skeleton
    squared_d_lens = squared_norm(d, keepdims=True)
    p_next_lens = norm(p_next, axis=-1, keepdims=True) + 1e-6
    p_prev_lens = norm(p_prev, axis=-1, keepdims=True) + 1e-6

    # If the length of d is zero, that means the point is co-linear with that segment of the skeleton,
    # and therefore no magnetic field is generated.
    ε = 1e-6

    d_scale = np.where(squared_d_lens > ε, 1 / (squared_d_lens + ε), 0)

    bs = d_scale * (np.cross(d, p_next) / p_next_lens - np.cross(d, p_prev) / p_prev_lens)

    b = bs.sum(axis=1) / (4 * np.pi)
    return b


def make_ring_mat(ring_position, ring_z_axis):
    """
    Make the 4x4 transform matrix that transforms from the canonical ring frame to the world frame.
    The canonical ring frame has +Z going up out of the ring

    Args:
        ring_position: [3] position of the center of the ring
        ring_z_axis: [3] unit vector pointing up out of the ring

    Returns: [4, 4] transform matrix

    """
    rand = np.random.rand(3)
    # project rand to its component in the plane of the ring
    ring_y_axis = rand - np.dot(rand, ring_z_axis) * ring_z_axis
    ring_y_axis /= np.linalg.norm(ring_y_axis)
    ring_x_axis = np.cross(ring_y_axis, ring_z_axis)
    ring_mat = np.eye(4)
    ring_mat[:3, 0] = ring_x_axis
    ring_mat[:3, 1] = ring_y_axis
    ring_mat[:3, 2] = ring_z_axis
    ring_mat[:3, 3] = ring_position
    return ring_mat


def make_ring_skeleton(position, z_axis, radius, delta_angle=0.5):
    angles = np.arange(0, 2 * np.pi, delta_angle)
    angles = np.append(angles, 0)
    zeros = np.zeros_like(angles)
    ones = np.ones_like(angles)
    ring_skeleton_canonical = np.stack([radius * np.cos(angles), radius * np.sin(angles), zeros, ones], -1)
    ring_mat = make_ring_mat(position, z_axis)
    skeleton = (ring_skeleton_canonical @ ring_mat.T)[:, :3]
    return skeleton


def get_h_signature(path, skeleton):
    path_discretized = discretize_path(path)
    path_deltas = np.diff(path_discretized, axis=0)
    bs = skeleton_field_dir(skeleton, path_discretized[:-1])
    h = np.sum(np.sum(bs * path_deltas, axis=-1), axis=0)
    return h


def discretize_path(path, n=1000):
    path_deltas = np.diff(path, axis=0)
    delta_lengths = np.linalg.norm(path_deltas, axis=-1)
    delta_cumsums = np.cumsum(delta_lengths)
    delta_cumsums = np.insert(delta_cumsums, 0, 0)
    total_length = np.sum(delta_lengths)
    l_s = np.linspace(1e-3, total_length, n)
    i_s = np.searchsorted(delta_cumsums, l_s)
    discretized_path_points = path[i_s - 1] + (l_s - delta_cumsums[i_s - 1])[:, None] * path_deltas[i_s - 1] / delta_lengths[i_s - 1][:, None]
    discretized_path_points = np.insert(discretized_path_points, 0, path[0], axis=0)
    return discretized_path_points
