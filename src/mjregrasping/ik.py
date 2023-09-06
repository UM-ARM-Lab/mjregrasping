import mujoco
import numpy as np

from mjregrasping.physics import get_gripper_ctrl_indices

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
