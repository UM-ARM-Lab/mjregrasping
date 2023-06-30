from typing import Optional

import mujoco
import numpy as np
from numpy.linalg import norm

from mjregrasping.params import hp
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


def eq_sim_ik(tool_names, candidate_is_grasping, candidate_pos, phy_ik, viz: Optional[Viz] = None):
    for _ in range(10):
        mujoco.mj_step(phy_ik.m, phy_ik.d, nstep=25)
        # Check if the grasping grippers are near their targets
        reached = True
        for i in range(hp['n_g']):
            if candidate_is_grasping[i]:
                d = norm(phy_ik.d.site(tool_names[i]).xpos - candidate_pos[i])
                if d > 0.01:
                    reached = False
        if viz:
            viz.viz(phy_ik, is_planning=True)
        if reached:
            return True
    return False
