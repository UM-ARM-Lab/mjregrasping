import numpy as np

from mjregrasping.physics import Physics


def grasp_location_to_indices(grasp_locations, rope_body_indices):
    grasp_indices = grasp_locations * (1 + rope_body_indices.max() - rope_body_indices.min()) + rope_body_indices.min()
    grasp_indices = np.clip(grasp_indices, rope_body_indices.min(), rope_body_indices.max())
    grasp_indices = np.floor(grasp_indices).astype(int)
    return grasp_indices


def grasp_indices_to_locations(rope_body_indices, grasp_index):
    grasp_locations = (grasp_index - rope_body_indices.min()) / (1 + rope_body_indices.max() - rope_body_indices.min())
    return grasp_locations


def grasp_offset(grasp_index, grasp_location, rope_body_indices):
    grasp_locations_rounded = grasp_indices_to_locations(rope_body_indices, grasp_index)
    offset = grasp_location - grasp_locations_rounded
    return offset


def grasp_locations_to_indices_and_offsets(grasp_locations, rope_body_indices):
    grasp_indices = grasp_location_to_indices(grasp_locations, rope_body_indices)
    offsets = grasp_offset(grasp_indices, grasp_locations, rope_body_indices)
    return grasp_indices, offsets


def grasp_locations_to_indices_and_offsets_and_xpos(phy: Physics, grasp_locations):
    grasp_indices = grasp_location_to_indices(grasp_locations, phy.o.rope.body_indices)
    offsets = grasp_offset(grasp_indices, grasp_locations, phy.o.rope.body_indices)
    body_xpos = phy.d.xpos[grasp_indices]
    body_xmat = phy.d.xmat[grasp_indices].reshape(-1, 3, 3)
    body_x_axis = body_xmat[:, :, 0]
    xpos = body_xpos + body_x_axis * offsets[:, None]
    return grasp_indices, offsets, xpos
