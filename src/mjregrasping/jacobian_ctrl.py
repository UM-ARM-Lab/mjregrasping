import mujoco
import numpy as np
from scipy.linalg import logm, block_diag

from mjregrasping.physics import Physics, get_q


def get_jacobian_ctrl(phy: Physics, tool_site, desired_mat_in_tool, v_in_tool, w_scale = 0.5, jnt_lim_avoidance=0.25):
    """
    Compute the joint velocities to move the tool site to the desired pose.
    Everything should be in tool frame.

    Args:
        phy: Physics
        tool_site: site object
        desired_mat_in_tool: Desired orientation
        v_in_tool: linear velocity in tool frame
        w_scale: scale factor for angular velocity
        jnt_lim_avoidance: scale factor for joint limit avoidance

    Returns:
        ctrl: joint velocities

    """
    w_in_tool = get_w_in_tool(desired_mat_in_tool, w_scale)
    twist_in_tool = np.concatenate([v_in_tool, w_in_tool])
    Jp = np.zeros((3, phy.m.nv))
    Jr = np.zeros((3, phy.m.nv))
    mujoco.mj_jacSite(phy.m, phy.d, Jp, Jr, tool_site.id)
    J_base = np.concatenate((Jp, Jr), axis=0)
    J_base = J_base[:, phy.m.actuator_trnid[:, 0]]
    # Transform J from base from to gripper frame
    tool2world_mat = phy.d.site_xmat[tool_site.id].reshape(3, 3)
    J_gripper = block_diag(tool2world_mat.T, tool2world_mat.T) @ J_base
    J_pinv = np.linalg.pinv(J_gripper)
    # use null-space projection to avoid joint limits
    current_q = get_q(phy)
    zero_vels = (phy.o.rd.q_home - current_q) * jnt_lim_avoidance
    warn_near_joint_limits(current_q, phy)
    ctrl = J_pinv @ twist_in_tool + (np.eye(phy.m.nu) - J_pinv @ J_gripper) @ zero_vels
    return ctrl, w_in_tool


def get_w_in_tool(desired_mat_in_tool, w_scale):
    # Now compute the angular velocity using matrix logarithm
    # https://youtu.be/WHn9xJl43nY?t=150
    W_in_tool = logm(desired_mat_in_tool).real
    w_in_tool = np.array([W_in_tool[2, 1], W_in_tool[0, 2], W_in_tool[1, 0]]) * w_scale
    return w_in_tool


def warn_near_joint_limits(current_q, phy):
    low = phy.m.actuator_actrange[:, 0]
    high = phy.m.actuator_actrange[:, 1]
    near_low = current_q < low + np.deg2rad(5)
    near_high = current_q > high - np.deg2rad(5)
    if np.any(near_low):
        i = np.where(near_low)[0][0]
        name_i = phy.o.robot.joint_names[i]
        q_i = current_q[i]
        limit_i = phy.m.actuator_actrange[i, 0]
        if 'gripper' not in name_i:
            print(f"WARNING: joint {name_i} is at {q_i}, near lower limit of {limit_i}!")
    elif np.any(near_high):
        i = np.where(near_high)[0][0]
        joint_i = phy.m.actuator_trnid[:, 0][i]
        name_i = phy.o.robot.joint_names[joint_i]

        q_i = current_q[i]
        limit_i = phy.m.actuator_actrange[i, 1]
        if 'gripper' not in name_i:
            print(f"WARNING: joint {name_i} is at {q_i}, near upper limit of {limit_i}!")
