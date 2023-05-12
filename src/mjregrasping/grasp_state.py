"""
A grasp "location" is from 0 to 1.
A grasp "index" is from n to m, and matches the body ids of the rope bodies in mujoco.
"""
import mujoco
import numpy as np

from mjregrasping.grasping import get_grasp_constraints


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


class GraspState:

    def __init__(self, rope_body_indices: np.ndarray, locations: np.ndarray):
        self.rope_body_indices = rope_body_indices
        self.locations = locations

    @staticmethod
    def from_mujoco(rope_body_indices: np.ndarray, m: mujoco.MjModel):
        eqs = get_grasp_constraints(m)
        locations = []
        for eq in eqs:
            if eq.active:
                b_id = m.body(eq.obj2id).id
                loc_floor = (b_id - rope_body_indices.min()) / (1 + rope_body_indices.max() - rope_body_indices.min())
                offset = np.linalg.norm(eq.data[3:6])
                loc = loc_floor + offset
            else:
                loc = 0
            locations.append(loc)
        return GraspState(rope_body_indices, np.array(locations))

    @property
    def is_grasping(self):
        return self.locations != 0

    @property
    def indices(self):
        return grasp_location_to_indices(self.locations, self.rope_body_indices)

    @property
    def offsets(self):
        return grasp_offset(self.indices, self.locations, self.rope_body_indices)

    def is_new(self, other):
        return other.is_grasping & ~self.is_grasping

    def is_diff(self, other):
        return (self.is_grasping == other.is_grasping) & (self.indices != other.indices)

    def needs_release(self, other):
        return self.is_grasping & ~other.is_grasping

    def set_is_grasping(self, is_grasping):
        self.locations = np.where(is_grasping, self.locations, np.zeros_like(self.locations))

    def __eq__(self, other):
        return np.all(self.indices == other.indices)

    def __repr__(self):
        return str(self.locations)

    def __str__(self):
        return str(self.locations)
