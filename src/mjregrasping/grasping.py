import mujoco
import numpy as np

from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets, grasp_indices_to_locations
from mjregrasping.physics import Physics


def get_is_grasping(phy):
    eqs = get_grasp_eqs(phy)
    is_grasping = np.array([phy.m.eq_active[eq.id] for eq in eqs])
    return is_grasping


def get_grasp_eqs(phy: Physics):
    return [phy.m.eq(eq_name) for eq_name in phy.o.rd.rope_grasp_eqs]


def activate_grasp(phy: Physics, name, loc):
    grasp_index, offset = grasp_locations_to_indices_and_offsets(loc, phy.o.rope.body_indices)
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


def get_finger_qs(phy: Physics):
    qs = [phy.d.qpos[phy.m.actuator(actuator_name).trnid[0]] for actuator_name in phy.o.rd.gripper_actuator_names]
    return np.stack(qs, axis=0)


def get_grasp_locs(phy: Physics):
    rope_length = get_rope_length(phy)
    locs = []
    for eq in get_grasp_eqs(phy):
        if eq.active:
            idx = int(eq.obj2id)
            offset = eq.data[3]
            loc = grasp_indices_to_locations(phy.o.rope.body_indices, idx) + (offset / rope_length)
            locs.append(loc)
        else:
            locs.append(-1)
    return np.array(locs)


def get_loc_idx_offset_xpos(phy: Physics):
    rope_length = get_rope_length(phy)
    for eq in get_grasp_eqs(phy):
        if eq.active:
            idx = eq.obj2id
            offset = eq.data[3]
            xmat = phy.d.xmat[idx].reshape(3, 3)
            xpos = phy.d.xpos[idx] + xmat[:, 0] * offset
            loc = grasp_indices_to_locations(phy.o.rope.body_indices, idx) + (offset / rope_length)
            yield loc, idx, offset, xpos
        continue


def get_rope_length(phy):
    rope_length = phy.m.geom(phy.o.rope.geom_indices[0]).size[1] * len(phy.o.rope.geom_indices) * 2
    return rope_length
