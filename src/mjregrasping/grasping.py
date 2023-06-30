import mujoco
import numpy as np

from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets


def get_is_grasping(m):
    eqs = get_grasp_eqs(m)
    is_grasping = np.array([m.eq_active[eq.id] for eq in eqs])
    return is_grasping


def get_grasp_eqs(m):
    return [m.eq('left'), m.eq('right')]


def activate_grasp(phy, name, loc, rope_body_indices):
    grasp_index, offset = grasp_locations_to_indices_and_offsets(loc, rope_body_indices)
    # Round the offset here so that the eqs are not too sensitive to small changes in the offset.
    # I was noticing that I got different cost when comparing `regrasp` and `untangle` and it turns out it was caused
    # by the offset being slightly different, since in one case the loc is sampled from code and in the other
    # it's typed in manually (to 3 decimal places).
    offset = round(offset, 2)
    offset_body = np.array([offset, 0, 0])
    grasp_eq = phy.m.eq(name)
    grasp_eq.obj2id = grasp_index
    grasp_eq.active = 1

    if grasp_eq.type == mujoco.mjtEq.mjEQ_CONNECT:
        grasp_eq.data[3:6] = offset_body
    elif grasp_eq.type == mujoco.mjtEq.mjEQ_WELD:
        obj1_pos = phy.d.body(grasp_eq.obj1id).xpos
        gripper_quat = phy.d.body(grasp_eq.obj1id).xquat
        b_pos = phy.d.body(grasp_eq.obj2id).xpos
        b_quat = phy.d.body(grasp_eq.obj2id).xquat

        neg_gripper_pos = np.zeros(3)
        neg_gripper_quat = np.zeros(4)
        mujoco.mju_negPose(neg_gripper_pos, neg_gripper_quat, obj1_pos, gripper_quat)

        g_b_pos = np.zeros(3)
        g_b_quat = np.zeros(4)
        mujoco.mju_mulPose(g_b_pos, g_b_quat, neg_gripper_pos, neg_gripper_quat, b_pos, b_quat)

        grasp_eq.data[0:3] = offset_body
        grasp_eq.data[3:6] = np.array([0, 0, 0.17])
        grasp_eq.data[6:10] = g_b_quat


def get_finger_qs(phy):
    leftgripper_q = phy.d.qpos[phy.m.actuator("leftgripper_vel").trnid[0]]
    rightgripper_q = phy.d.qpos[phy.m.actuator("rightgripper_vel").trnid[0]]
    return np.stack([leftgripper_q, rightgripper_q], axis=0)
