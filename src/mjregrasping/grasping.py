import mujoco
import numpy as np

from mjregrasping.grasp_state_utils import grasp_location_to_indices, grasp_offset, \
    grasp_locations_to_indices_and_offsets
from mjregrasping.physics import Physics


def get_grasp_indices(m):
    eqs = get_grasp_constraints(m)
    grasped_rope_body_ids = np.array([m.body(eq.obj2id).id if eq.active else -1 for eq in eqs])
    return grasped_rope_body_ids


def get_is_grasping(m):
    eqs = get_grasp_constraints(m)
    is_grasping = np.array([m.eq_active[eq.id] for eq in eqs])
    return is_grasping


def get_grasp_constraints(m):
    return [m.eq('left'), m.eq('right')]


def deactivate_eq(m, eq_name):
    eq_idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
    m.eq_active[eq_idx] = 0


def compute_eq_errors(phy: Physics):
    eq_errs = []
    for i in range(phy.m.neq):
        eq = phy.m.eq(i)
        if eq.active and eq.type == mujoco.mjtEq.mjEQ_CONNECT:
            eq_err = compute_eq_error(phy, eq)
            eq_errs.append(eq_err)
    return sum(eq_errs)


def compute_eq_error(phy, eq):
    b2 = phy.d.body(eq.obj2id)
    b1 = phy.d.body(eq.obj1id)
    b1_offset = eq.data[0:3]
    b2_offset = eq.data[3:6]
    b1_offset_in_world = np.zeros(3)
    mujoco.mju_trnVecPose(b1_offset_in_world, b1.xpos, b1.xquat, b1_offset)
    b2_offset_in_world = np.zeros(3)
    mujoco.mju_trnVecPose(b2_offset_in_world, b2.xpos, b2.xquat, b2_offset)
    eq_err = np.sum(np.square(b1_offset_in_world - b2_offset_in_world))
    return eq_err


def gripper_idx_to_eq_name(gripper_idx):
    return 'left' if gripper_idx == 0 else 'right'


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
    return leftgripper_q, rightgripper_q
