from copy import copy
from typing import Optional

import mujoco
import numpy as np

from mjregrasping.mujoco_objects import MjObjects


class Physics:
    m: mujoco.MjModel
    d: mujoco.MjData
    o: MjObjects

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, objects: Optional[MjObjects] = None):
        self.m = m
        self.d = d
        self.o = objects

    def __copy__(self):
        raise NotImplementedError("Use .copy_data() or .copy_all() to avoid ambiguity")

    def copy_data(self):
        new_phy = Physics(self.m, copy(self.d), self.o)
        # contact state is not copied, so we need to run forward to update it
        mujoco.mj_step1(new_phy.m, new_phy.d)
        return new_phy

    def copy_all(self):
        """ Much slower, since copying the model is slow """
        new_phy = Physics(copy(self.m), copy(self.d), self.o)
        # contact state is not copied, so we need to run forward to update it
        mujoco.mj_step1(new_phy.m, new_phy.d)
        return new_phy


def get_contact_forces(phy: Physics):
    contact_geoms = phy.d.contact.geom1
    contact_bodies = phy.m.geom_bodyid[contact_geoms]
    contact_forces = phy.d.cfrc_ext[contact_bodies]
    return contact_forces


def get_total_contact_force(phy: Physics):
    return np.sum(np.linalg.norm(get_contact_forces(phy), axis=-1))
