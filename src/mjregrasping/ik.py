from typing import Optional, Callable

import mujoco
import numpy as np
from numpy.linalg import norm

from mjregrasping.geometry import pairwise_squared_distances
from mjregrasping.goal_funcs import get_contact_cost
from mjregrasping.params import hp
from mjregrasping.physics import Physics, get_gripper_ctrl_indices
from mjregrasping.rollout import control_step, no_results
from mjregrasping.viz import Viz

BIG_PENALTY = 100


def position_jacobian(phy, body_idx, target_position):
    J = full_body_jacobian(phy, body_idx)
    J_T = J.T
    # TODO: account for ee_offset here
    current_ee_pos = phy.d.body(body_idx).xpos
    ee_vel_p = target_position - current_ee_pos  # Order of subtraction matters here!
    ee_vel = np.concatenate((ee_vel_p, np.zeros(3)), axis=0)
    eps = 1e-3
    ctrl = J_T @ np.linalg.solve(J @ J_T + eps * np.eye(6), ee_vel)
    gripper_ctrl_indices = get_gripper_ctrl_indices(phy)
    ctrl[gripper_ctrl_indices] = 0  # ignore the grippers
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


def full_body_jacobian(phy, body_idx):
    # compute the "end-effector" jacobian which relates the joint velocities to the end-effector velocity
    # where the ee_offset specifies the location/offset from the end-effector body
    Jp = np.zeros((3, phy.m.nv))
    Jr = np.zeros((3, phy.m.nv))
    mujoco.mj_jacBody(phy.m, phy.d, Jp, Jr, body_idx)
    J = np.concatenate((Jp, Jr), axis=0)
    J = J[:, phy.m.actuator_trnid[:, 0]]
    return J


def eq_sim_ik(candidate_is_grasping, candidate_pos, phy_ik: Physics, viz: Optional[Viz] = None,
              result_func: Callable = no_results):
    # Next activate the eqs between the grippers and the world to drag them to candidate_pos
    for i, gripper_to_world_eq_name in enumerate(phy_ik.o.rd.world_gripper_eqs):
        if candidate_is_grasping[i]:
            gripper_to_world_eq = phy_ik.m.eq(gripper_to_world_eq_name)
            gripper_to_world_eq.active = 1
            gripper_to_world_eq.data[3:6] = candidate_pos[i]
            gripper_to_world_eq.solref[0] = 1.0  # start with a really soft constraint

    results = []
    for t in range(hp['sim_ik_nstep']):
        for i, gripper_to_world_eq_name in enumerate(phy_ik.o.rd.world_gripper_eqs):
            if candidate_is_grasping[i]:
                gripper_to_world_eq = phy_ik.m.eq(gripper_to_world_eq_name)
                gripper_to_world_eq.solref[0] *= hp['sim_ik_solref_decay'] + hp['sim_ik_min_solref']
        control_step(phy_ik, np.zeros(phy_ik.m.nu), sub_time_s=hp['sim_ik_sub_time_s'])
        results.append(result_func(phy_ik))
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
            return True, results
    return False, results


def get_reachability_cost(phy_before, phy_after, reached, locs, is_grasping):
    contact_cost_before = get_contact_cost(phy_before)
    contact_cost_after = get_contact_cost(phy_after)
    new_contact_cost = contact_cost_after - contact_cost_before

    # Penalize the distance between locations if grasping with multiple grippers
    dists = pairwise_squared_distances(locs[:, None], locs[:, None])
    is_grasping_mat = np.tile(is_grasping, (len(is_grasping), 1))
    is_pair_grasping = is_grasping_mat * is_grasping_mat.T
    valid_grasp_dists = np.triu(dists * is_pair_grasping, k=1)
    nearby_locs_cost = np.sum(valid_grasp_dists) * hp['nearby_locs_weight']
    return new_contact_cost + nearby_locs_cost if reached else BIG_PENALTY


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
