import mujoco

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.mujoco_visualizer import MujocoVisualizer, RVizPublishers


def initialize(node_name, xml_path):
    rospy.init_node(node_name)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    tfw = TF2Wrapper()
    mjviz = MujocoVisualizer(tfw)

    mujoco.mj_forward(model, data)
    mjviz.viz(model, data)
    viz_pubs = RVizPublishers(tfw)

    return model, data, mjviz, viz_pubs


def activate_eq(model, eq_name):
    eq_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
    model.eq_active[eq_idx] = 1
    model.eq_data[eq_idx][3:6] = 0
