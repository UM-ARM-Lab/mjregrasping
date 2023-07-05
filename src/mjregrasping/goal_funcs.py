""" Free functions used by the goals """
import mujoco
import numpy as np
from numpy.linalg import norm

from mjregrasping.geometry import point_to_line_segment, pairwise_squared_distances
from mjregrasping.ik import jacobian_ik_is_reachable
from mjregrasping.mujoco_objects import Objects
from mjregrasping.params import hp
from mjregrasping.physics import Physics


def get_contact_cost(phy: Physics):
    # TODO: use SDF to compute near-contact cost to avoid getting too close
    # doing the contact cost calculation here means we don't need to return the entire data.contact array,
    # which makes things simpler and possibly faster, since this operation can't be easily vectorized.
    all_obstacle_geoms = phy.o.obstacle.geom_names + ['floor']
    contact_cost = 0
    for contact in phy.d.contact:
        geom_name1 = phy.m.geom(contact.geom1).name
        geom_name2 = phy.m.geom(contact.geom2).name
        if (geom_name1 in all_obstacle_geoms and geom_name2 in phy.o.val_collision_geom_names) or \
                (geom_name2 in all_obstacle_geoms and geom_name1 in phy.o.val_collision_geom_names) or \
                val_self_collision(geom_name1, geom_name2, phy.o):
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


def get_results_common(phy: Physics):
    joint_indices_for_actuators = phy.m.actuator_trnid[:, 0]
    joint_positions = phy.d.qpos[joint_indices_for_actuators]
    contact_cost = get_contact_cost(phy)
    tools_pos = get_tool_points(phy)
    is_unstable = phy.d.warning.number.sum() > 0
    return tools_pos, joint_positions, contact_cost, is_unstable


def get_tool_points(phy):
    left_tool_pos = phy.d.site('left_tool').xpos
    right_tool_pos = phy.d.site('right_tool').xpos
    return np.stack([left_tool_pos, right_tool_pos], 0)


def get_rope_points(phy):
    rope_points = phy.d.xpos[phy.o.rope.body_indices]
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


def get_finger_cost(finger_qs, desired_is_grasping):
    # Double the q_open so we are encouraged to open a lot more than the minimum required to release the rope
    desired_finger_qs = np.array(
        [hp['finger_q_closed'] if is_g_i else 2 * hp['finger_q_open'] for is_g_i in desired_is_grasping])
    finger_cost = (np.sum(np.abs(finger_qs - desired_finger_qs), axis=-1))
    return finger_cost


def get_regrasp_costs(finger_qs, is_grasping, desired_locs, regrasp_xpos, tools_pos, rope_points):
    """

    Args:
        finger_qs: Finger joint angle
        is_grasping:  Whether the gripper is grasping
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
    regrasp_finger_cost = get_finger_cost(finger_qs, desired_is_grasping) * hp['finger_weight']
    regrasp_dists = norm(regrasp_xpos - tools_pos, axis=-1)
    needs_grasp = desired_is_grasping * (1 - is_grasping)
    regrasp_pos_cost = np.sum(regrasp_dists * needs_grasp, -1) * hp['regrasp_weight']

    dists = pairwise_squared_distances(regrasp_xpos, rope_points)
    min_dist = np.min(dists)
    regrasp_near_cost = min_dist * hp['regrasp_near_weight']
    return regrasp_finger_cost, regrasp_pos_cost, regrasp_near_cost


def ray_based_reachability(candidates_xpos, phy, tools_pos, max_d=0.7):
    # by having group 1 set to 0, we exclude the rope and grippers/fingers
    # FIXME: this reachability check is not accurate!
    include_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    is_reachable = np.zeros([len(tools_pos), len(candidates_xpos)], dtype=bool)
    for i, tool_pos in enumerate(tools_pos):
        for j, xpos in enumerate(candidates_xpos):
            out_geomids = np.array([-1], dtype=np.int32)
            candidate_to_tool = (tool_pos - xpos)
            # print(i, j, candidate_to_tool, out_geomids)
            if norm(candidate_to_tool) > max_d:
                # print("not reachable because the candidate is too far away")
                is_reachable[i, j] = False
                continue
            d = mujoco.mj_ray(phy.m, phy.d, xpos, candidate_to_tool, include_groups, 1, -1, out_geomids)
            if d > max_d or out_geomids[0] == -1:
                # print("reachable because either there was no collision, or the collision was far away")
                is_reachable[i, j] = True
            else:
                # print("not reachable because there was a collision and it was close")
                is_reachable[i, j] = False
    return is_reachable


def ik_based_reachability(candidates_xpos, phy, tools_pos):
    is_reachable = np.zeros([len(tools_pos), len(candidates_xpos)], dtype=bool)
    from time import perf_counter
    t0 = perf_counter()
    for i, tool_pos in enumerate(tools_pos):
        tool_body_idx = phy.m.body('left_finger_pad').id if i == 0 else phy.m.body('right_finger_pad').id
        for j, xpos in enumerate(candidates_xpos):
            is_reachable[i, j] = jacobian_ik_is_reachable(phy, tool_body_idx, xpos, pos_tol=0.03)
    print(f'dt: {perf_counter() - t0:.4f}')
    return is_reachable
