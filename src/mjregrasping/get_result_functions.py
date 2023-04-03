import mujoco

def get_left_tool_pos_and_contact_cost(model, data):
    contact_cost = 0
    for contact in data.contact:
        geom_name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom_name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        if geom_name1 == 'obstacle' or geom_name2 == 'obstacle':
            contact_cost += 1
    return data.site_xpos[model.site('left_tool').id], contact_cost
