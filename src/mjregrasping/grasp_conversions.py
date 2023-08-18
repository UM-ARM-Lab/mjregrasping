import numpy as np

from mjregrasping.physics import Physics
from mjregrasping.rope_length import get_rope_length


def grasp_locations_to_is_grasping(grasp_locations):
    return grasp_locations != -1


def grasp_location_to_indices(grasp_locations, rope_body_indices):
    grasp_indices = grasp_locations * (1 + rope_body_indices.max() - rope_body_indices.min()) + rope_body_indices.min()
    grasp_indices = np.clip(grasp_indices, rope_body_indices.min(), rope_body_indices.max())
    grasp_indices = np.floor(grasp_indices).astype(int)
    return grasp_indices


def grasp_indices_to_locations(rope_body_indices, grasp_index):
    grasp_locations = (grasp_index - rope_body_indices.min()) / (1 + rope_body_indices.max() - rope_body_indices.min())
    return grasp_locations


def grasp_offset(grasp_index, grasp_location, phy: Physics):
    grasp_locations_rounded = grasp_indices_to_locations(phy.o.rope.body_indices, grasp_index)
    offset = (grasp_location - grasp_locations_rounded) * get_rope_length(phy)
    return offset


def grasp_locations_to_indices_and_offsets(grasp_locations, phy: Physics):
    grasp_indices = grasp_location_to_indices(grasp_locations, phy.o.rope.body_indices)
    offsets = grasp_offset(grasp_indices, grasp_locations, phy)
    return grasp_indices, offsets


def grasp_locations_to_indices_and_offsets_and_xpos(phy: Physics, grasp_locations):
    grasp_indices, offsets = grasp_locations_to_indices_and_offsets(grasp_locations, phy)
    xpos = body_plus_offset(phy, grasp_indices, offsets)
    return grasp_indices, offsets, xpos


def grasp_locations_to_xpos(phy: Physics, grasp_locations):
    if isinstance(grasp_locations, list):
        grasp_locations = np.array(grasp_locations)
    grasp_indices, offsets = grasp_locations_to_indices_and_offsets(grasp_locations, phy)
    xpos = body_plus_offset(phy, grasp_indices, offsets)
    return xpos


def body_plus_offset(phy, body_indices, offsets):
    body_xpos = phy.d.xpos[body_indices]
    body_xmat = phy.d.xmat[body_indices].reshape(-1, 3, 3)
    body_x_axis = body_xmat[:, :, 0]
    xpos = body_xpos + body_x_axis * offsets[:, None]
    return xpos


def make_full_locs(locs_where_grasping, is_grasping):
    candidate_locs = []
    j = 0
    for is_grasping_i in is_grasping:
        if is_grasping_i:
            candidate_locs.append(locs_where_grasping[j])
            j += 1
        else:
            candidate_locs.append(-1)
    candidate_locs = np.array(candidate_locs)
    return candidate_locs
