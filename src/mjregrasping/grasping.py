import mujoco
import numpy as np

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


def activate_eq(m, eq_name):
    eq = m.eq(eq_name)
    eq.active = 1
    eq.data[3:6] = 0


def deactivate_eq(m, eq_name):
    eq_idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
    m.eq_active[eq_idx] = 0


def compute_eq_errors(phy: Physics):
    eq_errs = []
    for i in range(phy.m.neq):
        eq = phy.m.eq(i)
        if eq.active:
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
    eq_err = np.linalg.norm(b1_offset_in_world - b2_offset_in_world)
    return eq_err
