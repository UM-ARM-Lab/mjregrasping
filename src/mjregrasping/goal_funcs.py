""" Free functions used by the goals """
import logging

import numpy as np
from numpy.linalg import norm

from mjregrasping.body_with_children import Objects
from mjregrasping.geometry import point_to_line_segment
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.ring_utils import make_ring_mat

logger = logging.getLogger(f'rosout.{__name__}')


def compute_threading_dir(ring_position, ring_z_axis, R, p, I=1, mu=1):
    """
    Compute the direction of a virtual magnetic field that will pull the rope through the ring.

    Args:
        ring_position: [3] The position of the origin of the ring in world coordinates
        ring_z_axis: [3] The axis point out up and out of the ring in world coordinates
        R: [1] radius of the ring in meters.
        p: [b, 3] the positions at which you want to compute the field
        I: [1] current, higher means stronger field. If you only care about direction, using 1 is fine.
        mu: [1] another scalar that controls the strength of the field. If you only care about direction, using 1 is fine.

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
    db = I * R * cross_product / np.linalg.norm(dp_repeated, axis=-1, keepdims=True) ** 3
    b = np.sum(db, axis=-2)
    b *= mu / (4 * np.pi)

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


def get_contact_cost(phy: Physics, objects: Objects):
    # TODO: use SDF to compute near-contact cost to avoid getting too close
    # doing the contact cost calculation here means we don't need to return the entire data.contact array,
    # which makes things simpler and possibly faster, since this operation can't be easily vectorized.
    all_obstacle_geoms = objects.obstacle.geom_names + ['floor']
    contact_cost = 0
    for contact in phy.d.contact:
        geom_name1 = phy.m.geom(contact.geom1).name
        geom_name2 = phy.m.geom(contact.geom2).name
        if (geom_name1 in all_obstacle_geoms and geom_name2 in objects.val_collision_geom_names) or \
                (geom_name2 in all_obstacle_geoms and geom_name1 in objects.val_collision_geom_names) or \
                val_self_collision(geom_name1, geom_name2, objects):
            contact_cost += 1
    max_expected_contacts = 6.0
    contact_cost /= max_expected_contacts
    # clamp to be between 0 and 1, and more sensitive to just a few contacts
    contact_cost = min(np.power(contact_cost, hp['contact_exponent']), hp['max_contact_cost']) * hp['contact_cost']
    return contact_cost


def val_self_collision(geom_name1, geom_name2, objects: Objects):
    return geom_name1 in objects.val_self_collision_geom_names and geom_name2 in objects.val_self_collision_geom_names


def get_action_cost(joint_positions):
    action_cost = np.sum(np.abs(joint_positions[..., 1:] - joint_positions[..., :-1]), axis=-1)
    action_cost *= hp['action']
    return action_cost


def get_results_common(objects: Objects, phy: Physics):
    joint_indices_for_actuators = phy.m.actuator_trnid[:, 0]
    joint_positions = phy.d.qpos[joint_indices_for_actuators]
    contact_cost = get_contact_cost(phy, objects)
    left_tool_pos = phy.d.site('left_tool').xpos
    right_tool_pos = phy.d.site('right_tool').xpos
    is_unstable = phy.d.warning.number.sum() > 0
    return left_tool_pos, right_tool_pos, joint_positions, contact_cost, is_unstable


def get_rope_points(phy, rope_body_indices):
    rope_points = np.array([phy.d.xpos[rope_body_idx] for rope_body_idx in rope_body_indices])
    return rope_points


def get_nearest_body_idx_and_offset(phy, body_indices, tool_pos):
    """
    Find the body index and offset of the body closest to the tool_pos.
    We can do this naively by looping over the bodies,
    treating it as a line-segment in 3D, computing the closest point on the line segment to the tool_pos,
    computing the offset, and taking the minimum.

    Args:
        phy:
        body_indices:
        tool_pos:

    Returns:

    """
    min_cost = np.inf
    min_body_idx = None
    min_offset = None
    # TODO: batch this
    for body_idx in body_indices[:-1]:
        body_pos = phy.d.xpos[body_idx]
        body_pos_next = phy.d.xpos[body_idx + 1]
        nearest_point, _ = point_to_line_segment(body_pos, body_pos_next, tool_pos)
        offset = np.linalg.norm(nearest_point - body_pos)

        dist = np.linalg.norm(tool_pos - nearest_point)
        cost = dist + offset * 0.01  # prefer the body with the smallest offset
        if cost < min_cost:
            min_cost = cost
            min_body_idx = body_idx
            min_offset = offset

    return min_body_idx, min_offset
