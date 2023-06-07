from copy import copy

import numpy as np


def get_left_tool_pos_and_contact_cost(phy):
    contact_cost = 0
    for contact in phy.d.contact:
        geom_name1 = phy.m.geom(contact.geom1).name
        geom_name2 = phy.m.geom(contact.geom2).name
        if geom_name1 == 'obstacle' or geom_name2 == 'obstacle':
            contact_cost += 1
    return phy.d.site_xpos[phy.m.site('left_tool').id], contact_cost


def get_q_current(phy):
    # NOTE: if I used "sensors" instead of "qpos", it might clearer because I could just omit the sensor
    #  for the mimic'd gripper joint.
    deduplicated_indices = np.array([0, 1,
                                     2, 3, 4, 5, 6, 7, 8,
                                     9,
                                     11, 12, 13, 14, 15, 16, 17,
                                     18])
    return copy(phy.d.qpos[deduplicated_indices])
