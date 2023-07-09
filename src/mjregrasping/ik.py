from typing import Optional

import mujoco
import numpy as np
from numpy.linalg import norm

from mjregrasping.geometry import pairwise_squared_distances
from mjregrasping.goal_funcs import get_contact_cost
from mjregrasping.grasping import get_grasp_eqs
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.viz import Viz


def jacobian_ik_is_reachable(phy, body_idx, target_point, n_steps=100, pos_tol=0.005):
    for i in range(n_steps):
        ctrl, error = position_jacobian(phy, body_idx, target_point)
        np.copyto(phy.d.ctrl, ctrl)
        mujoco.mj_step(phy.m, phy.d, nstep=25)

        if error < pos_tol:
            return True

    return False


def position_jacobian(phy, body_idx, target_position, ee_offset=np.zeros(3)):
    # compute the "end-effector" jacobian which relates the joint velocities to the end-effector velocity
    # where the ee_offset specifies the location/offset from the end-effector body
    Jp = np.zeros((3, phy.m.nv))
    Jr = np.zeros((3, phy.m.nv))
    mujoco.mj_jac(phy.m, phy.d, Jp, Jr, ee_offset, body_idx)
    J = np.concatenate((Jp, -Jr), axis=0)
    deduplicated_indices = np.array([0, 1,
                                     2, 3, 4, 5, 6, 7, 8,
                                     9,
                                     11, 12, 13, 14, 15, 16, 17,
                                     18])
    J = J[:, deduplicated_indices]
    J_T = J.T
    # TODO: account for ee_offset here
    current_ee_pos = phy.d.body(body_idx).xpos
    ee_vel_p = target_position - current_ee_pos  # Order of subtraction matters here!
    ee_vel = np.concatenate((ee_vel_p, np.zeros(3)), axis=0)
    eps = 1e-3
    ctrl = J_T @ np.linalg.solve(J @ J_T + eps * np.eye(6), ee_vel)
    ctrl[9] = 0  # ignore the gripper joint
    ctrl[17] = 0  # ignore the gripper joint
    # TODO: use nullspace to respect joint limits by trying to move towards the home configuration

    # rescale to respect velocity limits
    vmin = phy.m.actuator_ctrlrange[:, 0]
    vmax = phy.m.actuator_ctrlrange[:, 1]
    if np.any(ctrl > vmax):
        offending_joint = np.argmax(ctrl)
        ctrl = ctrl / np.max(ctrl) * vmax[offending_joint]
    if np.any(ctrl < vmin):
        offending_joint = np.argmin(ctrl)
        ctrl = ctrl / np.min(ctrl) * vmin[offending_joint]

    position_error = np.linalg.norm(target_position - phy.d.body(body_idx).xpos)

    return ctrl, position_error


def eq_sim_ik(candidate_is_grasping, candidate_pos, phy_ik: Physics, viz: Optional[Viz] = None):
    # TODO: what if we start with a large solref[0] and decrease it? to make the constraint harder over time
    for _ in range(10):
        mujoco.mj_step(phy_ik.m, phy_ik.d, nstep=25)
        # Check if the grasping grippers are near their targets
        reached = True
        for i, (tool_name, candidate_pos_i) in enumerate(zip(phy_ik.o.rd.tool_sites, candidate_pos)):
            if candidate_is_grasping[i]:
                d = norm(phy_ik.d.site(tool_name).xpos - candidate_pos_i)
                if d > phy_ik.o.rd.ik_tol:
                    reached = False
        if viz:
            viz.viz(phy_ik, is_planning=True)
        if reached:
            return True
    return False


def get_reachability_cost(phy_before, phy_after, reached, locs, is_grasping):
    q_before = phy_before.d.qpos[phy_before.o.robot.qpos_indices]
    q_after = phy_after.d.qpos[phy_after.o.robot.qpos_indices]
    contact_cost_before = get_contact_cost(phy_before)
    contact_cost_after = get_contact_cost(phy_after)
    new_contact_cost = contact_cost_after - contact_cost_before

    reachability_cost = np.linalg.norm(q_after - q_before)
    # Penalize the distance between locations if grasping
    dists = pairwise_squared_distances(locs[:, None], locs[:, None])
    is_grasping_mat = np.tile(is_grasping, (len(is_grasping), 1))
    is_pair_grasping = is_grasping_mat * is_grasping_mat.T
    valid_grasp_dists = np.triu(dists * is_pair_grasping, k=1)
    nearby_locs_cost = np.sum(valid_grasp_dists) * hp['nearby_locs_weight']
    return new_contact_cost + reachability_cost + nearby_locs_cost if reached else HARD_CONSTRAINT_PENALTY


HARD_CONSTRAINT_PENALTY = 1e3
IK_OFFSET = np.array([0, 0, 0.145])


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


def create_eq_and_sim_ik(phy: Physics, candidate_is_grasping, candidate_idx,
                         candidate_pos, viz):
    phy_ik = phy.copy_all()
    grasp_eqs = get_grasp_eqs(phy_ik)
    ik_targets = {}
    for i in range(phy_ik.o.rd.n_g):
        tool_name = phy_ik.o.rd.tool_sites[i]
        gripper_to_world_eq_name = phy_ik.o.rd.world_gripper_eqs[i]
        eq = grasp_eqs[i]
        # Set eq_active and qpos depending on the grasp
        if candidate_is_grasping[i]:
            eq.active = 1
            eq.obj2id = candidate_idx[i]
            ik_targets[tool_name] = candidate_pos[i] - IK_OFFSET
            gripper_to_world_eq = phy_ik.m.eq(gripper_to_world_eq_name)
            gripper_to_world_eq.active = 1
            gripper_to_world_eq.data[3:6] = candidate_pos[i]
        else:
            eq.active = 0
    # Estimate reachability using eq constraints in mujoco
    reached = eq_sim_ik(candidate_is_grasping, candidate_pos, phy_ik, viz=viz)
    return phy_ik, reached
