from copy import copy
from dataclasses import dataclass

import mujoco


@dataclass
class Physics:
    m: mujoco.MjModel
    d: mujoco.MjData

    def __copy__(self):
        # only copy data, not model, since the model is usually treated as "read-only"
        new_d = copy(self.d)
        new_phy = Physics(self.m, new_d)
        # contact state is not copied, so we need to run forward to update it
        mujoco.mj_step1(new_phy.m, new_phy.d)
        return new_phy
