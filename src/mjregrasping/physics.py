from copy import copy

import mujoco

from mjregrasping.mujoco_objects import Objects


class Physics:
    m: mujoco.MjModel
    d: mujoco.MjData
    o: Objects

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, obstacle_name):
        self.m = m
        self.d = d
        self.obstacle_name = obstacle_name
        self.o = Objects(m, obstacle_name)

    def __copy__(self):
        raise NotImplementedError("Use .copy_data() or .copy_all() to avoid ambiguity")

    def copy_data(self):
        new_phy = Physics(self.m, copy(self.d), self.obstacle_name)
        # contact state is not copied, so we need to run forward to update it
        mujoco.mj_step1(new_phy.m, new_phy.d)
        return new_phy

    def copy_all(self):
        """ Much slower, since copying the model is slow """
        new_phy = Physics(copy(self.m), copy(self.d), self.obstacle_name)
        # contact state is not copied, so we need to run forward to update it
        mujoco.mj_step1(new_phy.m, new_phy.d)
        return new_phy


if __name__ == '__main__':
    m = mujoco.MjModel.from_xml_path("models/untangle_scene.xml")
    d = mujoco.MjData(m)
    phy = Physics(m, d, "computer_rack")
    from time import perf_counter

    t0 = perf_counter()
    for i in range(100):
        phy2 = phy.copy_data()
    print(f'dt: {perf_counter() - t0:.4f}')

    t0 = perf_counter()
    for i in range(100):
        phy2 = phy.copy_all()
    print(f'dt: {perf_counter() - t0:.4f}')