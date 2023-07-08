from copy import copy
from typing import Optional

import mujoco

from mjregrasping.mujoco_objects import Objects


class Physics:
    m: mujoco.MjModel
    d: mujoco.MjData
    o: Objects

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, objects: Optional[Objects] = None):
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
