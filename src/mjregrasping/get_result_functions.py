from copy import copy

import mujoco


def get_left_tool_pos_and_contact_cost(phy):
    contact_cost = 0
    for contact in phy.d.contact:
        geom_name1 = phy.m.geom(contact.geom1).name
        geom_name2 = phy.m.geom(contact.geom2).name
        if geom_name1 == 'obstacle' or geom_name2 == 'obstacle':
            contact_cost += 1
    return phy.d.site_xpos[phy.m.site('left_tool').id], contact_cost


def get_q_current(phy):
    # FIXME: don't hardcode the indices
    return copy(phy.d.qpos[:20])
