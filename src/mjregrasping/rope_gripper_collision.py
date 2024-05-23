from itertools import chain

from mjregrasping.physics import Physics


def disable_rope_gripper_collisions(phy: Physics):
    for geom_name in list(chain(*phy.o.rd.gripper_geom_names)) + phy.o.rope.geom_names:
        phy.m.geom(geom_name).contype = 1
        phy.m.geom(geom_name).conaffinity = 2
