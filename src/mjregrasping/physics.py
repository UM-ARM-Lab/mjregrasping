from copy import copy
from dataclasses import dataclass

import mujoco


@dataclass
class Physics:
    m: mujoco.MjModel
    d: mujoco.MjData

    def __copy__(self):
        raise NotImplementedError("Use .copy_data() or .copy_all() to avoid ambiguity")

    def copy_data(self):
        # TODO: we can skip some fields like xfrc_applied which are quite large
        new_phy = Physics(self.m, copy(self.d))
        # contact state is not copied, so we need to run forward to update it
        mujoco.mj_step1(new_phy.m, new_phy.d)
        return new_phy

    def copy_all(self):
        """ Much slower, since copying the model is slow """
        # we can skip textures?
        new_phy = Physics(copy(self.m), copy(self.d))
        # contact state is not copied, so we need to run forward to update it
        mujoco.mj_step1(new_phy.m, new_phy.d)
        return new_phy


if __name__ == '__main__':
    m = mujoco.MjModel.from_xml_path("models/untangle_scene.xml")
    d = mujoco.MjData(m)
    phy = Physics(m, d)
    from time import perf_counter

    t0 = perf_counter()
    for i in range(100):
        phy2 = phy.copy_data()
    print(f'dt: {perf_counter() - t0:.4f}')

    t0 = perf_counter()
    for i in range(100):
        phy2 = phy.copy_all()
    print(f'dt: {perf_counter() - t0:.4f}')

    copy(phy)
