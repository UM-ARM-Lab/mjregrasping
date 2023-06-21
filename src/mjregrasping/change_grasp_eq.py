import numpy as np

from mjregrasping.grasp_state_utils import grasp_locations_to_indices_and_offsets


def change_eq(m, eq_name, grasp_location, rope_body_indices):
    """ this will not preserve the initial offset between obj1 and obj2, it will teleport/pull obj2 to obj1. """
    grasp_index, offset = grasp_locations_to_indices_and_offsets(grasp_location, rope_body_indices)
    offset = round(offset, 2)
    eq = m.eq(eq_name)
    eq.obj2id = grasp_index
    eq.active = 1
    eq.data[3:6] = np.array([offset, 0, 0])


