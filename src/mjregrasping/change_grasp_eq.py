import numpy as np

from mjregrasping.grasp_state import grasp_location_to_indices, grasp_offset


def change_eq(m, eq_name, grasp_location, rope_body_indices):
    grasp_index = grasp_location_to_indices(grasp_location, rope_body_indices)
    x_offset = grasp_offset(grasp_index, grasp_location, rope_body_indices)
    x_offset = round(x_offset, 2)
    eq = m.eq(eq_name)
    eq.obj2id = grasp_index
    eq.active = 1
    eq.data[3:6] = np.array([x_offset, 0, 0])
