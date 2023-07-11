import numpy as np

from mjregrasping.physics import Physics
from mjregrasping.rope_length import get_rope_length


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
    body_xpos = phy.d.xpos[grasp_indices]
    body_xmat = phy.d.xmat[grasp_indices].reshape(-1, 3, 3)
    body_x_axis = body_xmat[:, :, 0]
    xpos = body_xpos + body_x_axis * offsets[:, None]
    return grasp_indices, offsets, xpos


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


def sln_to_locs(best_sln, candidate_is_grasping):
    locs = []
    best_sln = list(best_sln.values())
    j = 0
    for is_grasping_i in candidate_is_grasping:
        if is_grasping_i:
            locs.append(best_sln[j])
            j += 1
        else:
            locs.append(-1)
    locs = np.array(locs)
    return locs
