import mujoco
import rerun as rr

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.rviz import MjRViz
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.viz import Viz


def initialize(node_name, xml_path):
    rr.init('mjregrasping')
    rr.connect()

    rospy.init_node(node_name)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # TODO: wrap send_transform in the Viz class, so we can do it to non-ros visualizers as well
    tfw = TF2Wrapper()
    mjviz = MjRViz(xml_path, tfw)

    mujoco.mj_forward(m, d)
    mjviz.viz(m, d)

    return m, d, Viz(rviz=mjviz, mjrr=MjReRun())
