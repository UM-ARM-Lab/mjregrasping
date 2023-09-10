import mujoco
import numpy as np

from mjregrasping.physics import Physics


def compute_total_eq_error(phy: Physics):
    eq_errs = []
    for eq_name in phy.o.rd.rope_grasp_eqs + ['attach']:
        eq = phy.m.eq(eq_name)
        if eq.active:
            eq_err = compute_eq_error(phy, eq)
            eq_errs.append(eq_err)
    return sum(eq_errs)


def compute_eq_error(phy, eq):
    b2 = phy.d.body(eq.obj2id)
    b1 = phy.d.body(eq.obj1id)
    if eq.type == mujoco.mjtEq.mjEQ_CONNECT:
        b1_offset = eq.data[0:3]
        b2_offset = eq.data[3:6]
    elif eq.type == mujoco.mjtEq.mjEQ_WELD:
        b1_offset = eq.data[3:6]
        b2_offset = eq.data[0:3]
    b1_offset_in_world = np.zeros(3)
    mujoco.mju_trnVecPose(b1_offset_in_world, b1.xpos, b1.xquat, b1_offset)
    b2_offset_in_world = np.zeros(3)
    mujoco.mju_trnVecPose(b2_offset_in_world, b2.xpos, b2.xquat, b2_offset)
    eq_err = np.sum(np.square(b1_offset_in_world - b2_offset_in_world))
    return eq_err
