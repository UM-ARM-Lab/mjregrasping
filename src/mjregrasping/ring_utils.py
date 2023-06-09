import numpy as np


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
