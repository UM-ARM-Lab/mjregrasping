import mujoco

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.mujoco_visualizer import MujocoVisualizer, RVizPublishers


def initialize(node_name, xml_path):
    rospy.init_node(node_name)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    tfw = TF2Wrapper()
    mjviz = MujocoVisualizer(tfw)
    mjviz.viz(model, data)

    viz_pubs = RVizPublishers(tfw)

    return model, data, mjviz, viz_pubs

