import mujoco
import numpy as np


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
