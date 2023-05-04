import mujoco

import rerun as rr
import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.mujoco_visualizer import MujocoVisualizer, RVizPublishers
from mjregrasping.params import Params


def initialize(node_name, xml_path):
    rr.init('mjregrasping')
    rr.connect()

    rospy.init_node(node_name)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # add a custom callback to define the sensor values for "external force"

    tfw = TF2Wrapper()
    mjviz = MujocoVisualizer(tfw)

    mujoco.mj_forward(m, d)
    mjviz.viz(m, d)
    viz_pubs = RVizPublishers(tfw)

    p = Params()

    return m, d, mjviz, viz_pubs, p


def activate_eq(m, eq_name):
    eq_idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
    m.eq_active[eq_idx] = 1
    m.eq_data[eq_idx][3:6] = 0
