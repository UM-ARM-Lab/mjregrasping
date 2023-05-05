import mujoco
import rerun as rr

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.rviz import MjRViz
from mjregrasping.params import Params
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.viz import Viz


def initialize(node_name, xml_path):
    rr.init('mjregrasping')
    rr.connect()

    rospy.init_node(node_name)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # add a custom callback to define the sensor values for "external force"

    # TODO: wrap send_transform in the Viz class so we can do it to non-ros visualizers as well
    tfw = TF2Wrapper()
    mjviz = MjRViz(tfw)

    mujoco.mj_forward(m, d)
    mjviz.viz(m, d)

    p = Params()

    return m, d, Viz(rviz=mjviz, mjrr=MjReRun(), tfw=tfw, p=p)
