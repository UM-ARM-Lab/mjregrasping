"""
A grasp "location" is from 0 to 1.
A grasp "index" is from n to m, and matches the body ids of the rope bodies in mujoco.
"""
from typing import Optional

import mujoco
import numpy as np

from mjregrasping.grasp_state_utils import grasp_location_to_indices, grasp_offset
from mjregrasping.grasping import get_grasp_constraints


class GraspState:

    def __init__(self, rope_body_indices: np.ndarray, locations: np.ndarray, is_grasping: Optional[np.ndarray] = None):
        self.rope_body_indices = rope_body_indices
        self.locations = locations
        self.is_grasping = is_grasping

    @staticmethod
    def from_mujoco(rope_body_indices: np.ndarray, m: mujoco.MjModel):
        eqs = get_grasp_constraints(m)
        locations = []
        is_grasping = []
        for eq in eqs:
            if eq.active:
                b_id = m.body(eq.obj2id).id
                loc_floor = (b_id - rope_body_indices.min()) / (1 + rope_body_indices.max() - rope_body_indices.min())
                offset = np.linalg.norm(eq.data[3:6])
                loc = loc_floor + offset
                is_grasping.append(True)
            else:
                loc = 0
                is_grasping.append(False)
            locations.append(loc)
        return GraspState(rope_body_indices, np.array(locations), np.array(is_grasping))

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

    def __eq__(self, other):
        return np.all(self.indices == other.indices) and np.all(self.is_grasping == other.is_grasping)

    def __repr__(self):
        return str(self.locations)

    def __str__(self):
        return str(self.locations)
