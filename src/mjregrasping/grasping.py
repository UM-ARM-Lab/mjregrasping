import numpy as np


def get_grasp_indices(model):
    eqs = get_grasp_constraints(model)
    grasped_rope_body_ids = np.array([model.body(eq.obj2id).id if eq.active else -1 for eq in eqs])
    return grasped_rope_body_ids


def get_is_grasping(model):
    eqs = get_grasp_constraints(model)
    is_grasping = np.array([model.eq_active[eq.id] for eq in eqs])
    return is_grasping


def get_grasp_constraints(model):
    return [model.eq('left'), model.eq('right')]
