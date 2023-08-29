""" Free functions used by the goals """
import numpy as np
from numpy.linalg import norm

from mjregrasping.geometry import point_to_line_segment, pairwise_squared_distances
from mjregrasping.grasp_conversions import grasp_locations_to_xpos
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.params import hp
from mjregrasping.physics import Physics


def get_nongrasping_rope_contact_cost(phy: Physics, desired_grasp_locs):
    contact_cost = 0
    for desired_grasp_loc, gripper_geom_names_i in zip(desired_grasp_locs, phy.o.rd.gripper_geom_names):
        if desired_grasp_loc != -1:
            continue
        for contact in phy.d.contact:
            geom_name1 = phy.m.geom(contact.geom1).name
            geom_name2 = phy.m.geom(contact.geom2).name
            if (geom_name1 in gripper_geom_names_i and geom_name2 in phy.o.rope.geom_names) or \
                    (geom_name2 in gripper_geom_names_i and geom_name1 in phy.o.rope.geom_names):
                contact_cost += 1
    contact_cost /= hp['max_expected_contacts']
    # clamp to be between 0 and 1, and more sensitive to just a few contacts
    contact_cost = min(np.power(contact_cost, hp['contact_exponent']), hp['max_contact_cost']) * hp['contact_cost']
    return contact_cost


def get_contact_cost(phy: Physics, verbose=False):
    # TODO: use SDF to compute near-contact cost to avoid getting too close
    # doing the contact cost calculation here means we don't need to return the entire data.contact array,
    # which makes things simpler and possibly faster, since this operation can't be easily vectorized.
    contact_cost = 0
    for contact in phy.d.contact:
        geom_name1 = phy.m.geom(contact.geom1).name
        geom_name2 = phy.m.geom(contact.geom2).name
        if is_valid_contact(phy, geom_name1, geom_name2):
            if verbose:
                print(f"Contact between {geom_name1} and {geom_name2}")
            contact_cost += 1
    contact_cost /= hp['max_expected_contacts']
    # clamp to be between 0 and 1, and more sensitive to just a few contacts
    contact_cost = min(np.power(contact_cost, hp['contact_exponent']), hp['max_contact_cost']) * hp['contact_cost']
    return contact_cost


def is_valid_contact(phy, geom_name1, geom_name2):
    all_obstacle_geoms = phy.o.obstacle.geom_names + ['floor']
    return (geom_name1 in all_obstacle_geoms and geom_name2 in phy.o.robot_collision_geom_names) or \
        (geom_name2 in all_obstacle_geoms and geom_name1 in phy.o.robot_collision_geom_names) or \
        val_self_collision(geom_name1, geom_name2, phy.o)


def val_self_collision(geom_name1, geom_name2, objects: MjObjects):
    return geom_name1 in objects.robot_self_collision_geom_names and geom_name2 in objects.robot_self_collision_geom_names


def get_action_cost(joint_positions):
    action_cost = np.sum(np.abs(joint_positions[..., 1:] - joint_positions[..., :-1]), axis=-1)
    action_cost *= hp['action']
    return action_cost


def get_results_common(phy: Physics):
    joint_indices_for_actuators = phy.m.actuator_trnid[:, 0]
    joint_positions = phy.d.qpos[joint_indices_for_actuators]
    contact_cost = get_contact_cost(phy)
    tools_pos = get_tool_points(phy)
    is_unstable = phy.d.warning.number.sum() > 0
    return tools_pos, joint_positions, contact_cost, is_unstable


def get_tool_points(phy):
    xpos = [phy.d.site(site_name).xpos for site_name in phy.o.rd.tool_sites]
    return np.stack(xpos, 0)


def get_rope_points(phy):
    rope_points = phy.d.xpos[phy.o.rope.body_indices]
    end_point = grasp_locations_to_xpos(phy, np.ones(1))[0]
    rope_points = np.concatenate([rope_points, [end_point]])
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


def get_keypoint(phy, body_idx, offset):
    """
    Get the position of the keypoint, which is specified by the body_idx and the offset along the Y axis of the body
    """
    xmat = phy.d.body(body_idx).xmat.reshape(3, 3)
    keypoint = phy.d.xpos[body_idx] + xmat[:, 0] * offset
    return keypoint


def get_regrasp_costs(finger_qs, is_grasping, current_grasp_locs, desired_locs, regrasp_xpos, tools_pos, rope_points):
    """

    Args:
        finger_qs: Finger joint angle
        is_grasping:  Whether the gripper is grasping
        current_grasp_locs: The current grasp locations ∈ [0-1]
        desired_locs:  Whether the gripper should grasp ∈ [0-1]
        regrasp_xpos: The 3d position in space corresponding to the regrasp_locs
        tools_pos: The current 3d position of the tool tips
        rope_points: The 3d position of all the rope points

    Returns:
        Costs for finger joint angles, distance to desired rope positions, and distance to nearest point on rope.
        The distance-to-nearest term is used to allow the controller to give up and just grasp a point near the desired
        regrasp position.

    """
    desired_is_grasping = desired_locs >= 0

    regrasp_dists = norm(regrasp_xpos - tools_pos, axis=-1)
    regrasp_pos_cost = np.sum(regrasp_dists * desired_is_grasping, -1) * hp['grasp_pos_weight']

    dists = pairwise_squared_distances(tools_pos, rope_points)
    min_dist = np.min(np.min(dists, axis=-1), axis=-1)
    regrasp_near_cost = min_dist * hp['grasp_near_weight']

    desired_open = check_should_be_open(current_grasp_locs, is_grasping, desired_locs, desired_is_grasping)
    # Double the q_open, so we are encouraged to open a lot more than the minimum required to release the rope
    desired_finger_qs = np.where(desired_open, 2 * hp['finger_q_open'], hp['finger_q_closed'])
    regrasp_finger_cost = (np.sum(np.abs(finger_qs - desired_finger_qs), axis=-1)) * hp['grasp_finger_weight']

    return regrasp_finger_cost, regrasp_pos_cost, regrasp_near_cost


def check_should_be_open(current_grasp_locs, current_is_grasping, desired_locs, desired_is_grasping):
    wrong_loc = ~locs_eq(current_grasp_locs, desired_locs)
    desired_open = np.logical_or(
        np.logical_and(np.logical_and(wrong_loc, desired_is_grasping), current_is_grasping),
        np.logical_not(desired_is_grasping)
    )
    return desired_open


def locs_eq(locs_a, locs_b):
    return abs(locs_a - locs_b) < hp['grasp_loc_diff_thresh']
