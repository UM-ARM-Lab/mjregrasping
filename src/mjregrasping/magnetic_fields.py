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


def compute_threading_dir(ring_position, ring_z_axis, R, p):
    """
    Compute the direction of a virtual magnetic field that will pull the rope through the ring.
    Assumes μ and I are 1.

    Args:
        ring_position: [3] The position of the origin of the ring in world coordinates
        ring_z_axis: [3] The axis point out up and out of the ring in world coordinates
        R: [1] radius of the ring in meters.
        p: [b, 3] the positions at which you want to compute the field

    Returns:
        [b, 3] the direction of the field at p

    """
    dx, x = compute_ring_vecs(ring_position, ring_z_axis, R)
    # We need a batch cross product
    b = p.shape[0]
    n = dx.shape[0]
    dx_repeated = np.repeat(dx[None], b, axis=0)
    x_repeated = np.repeat(x[None], b, axis=0)
    p_repeated = np.repeat(p[:, None], n, axis=1)
    dp_repeated = p_repeated - x_repeated
    cross_product = np.cross(dx_repeated, dp_repeated)
    db = R * cross_product / np.linalg.norm(dp_repeated, axis=-1, keepdims=True) ** 3
    b = np.sum(db, axis=-2)
    b *= 1 / (4 * np.pi)

    return b


def compute_ring_vecs(ring_position, ring_z_axis, R):
    """
    Compute the direction sub-quantities needed for compute_threading_dir

    Args:
        ring_position: [3] The position of the origin of the ring in world coordinates
        ring_z_axis: [3] The axis point out up and out of the ring in world coordinates
        R: [1] radius of the ring in meters.

    Returns:
        [n, 3] the direction of the conductor (ring) at each ring point
        [n, 3] the position of the ring points

    """
    delta_angle = 0.1
    angles = np.arange(0, 2 * np.pi, delta_angle)
    zeros = np.zeros_like(angles)
    ones = np.ones_like(angles)
    x = np.stack([R * np.cos(angles), R * np.sin(angles), zeros, ones], -1)
    ring_mat = make_ring_mat(ring_position, ring_z_axis)
    x = (x @ ring_mat.T)[:, :3]
    zeros = np.zeros_like(angles)
    dx = np.stack([-np.sin(angles), np.cos(angles), zeros], -1)
    # only rotate here
    ring_rot = ring_mat.copy()[:3, :3]
    dx = (dx @ ring_rot.T)
    dx = dx / np.linalg.norm(dx, axis=-1, keepdims=True)  # normalize
    return dx, x


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
