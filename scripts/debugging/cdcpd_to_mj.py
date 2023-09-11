#!/usr/bin/env python3

import mujoco
import numpy as np
import rerun as rr

import rospy
# noinspection PyUnresolvedReferences
import tf2_geometry_msgs
from arc_utilities import ros_init
from arc_utilities.listener import Listener
from arc_utilities.tf2wrapper import TF2Wrapper
from geometry_msgs.msg import PointStamped
from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.grasping import activate_grasp
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.physics import Physics
from mjregrasping.real_val import RealValCommander
from mjregrasping.set_up_real_scene import set_up_real_scene
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.scenarios import real_goal_sig, get_real_untangle_skeletons
from mjregrasping.val_dup import val_dedup
from mjregrasping.viz import make_viz, Viz
from srv import SetCDCPDState, SetCDCPDStateRequest
from visualization_msgs.msg import MarkerArray


@ros_init.with_ros("cdcpd_to_mj")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('untangle')
    rr.connect()

    scenario = real_goal_sig

    viz = make_viz(scenario)
    cdcpd_sub = Listener("/cdcpd_pred", MarkerArray)

    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
    d = mujoco.MjData(m)
    phy = Physics(m, d, MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))
    mujoco.mj_forward(m, d)
    viz.viz(phy, False)

    val_cmd = RealValCommander(phy)

    tfw = TF2Wrapper()

    mov = None
    set_up_real_scene(val_cmd, phy, viz)

    skeletons = get_real_untangle_skeletons(phy)
    viz.skeletons(skeletons)

    val_cmd.set_cdcpd_from_mj_rope(phy)

    val_cmd.pull_rope_towards_cdcpd(phy)

    for _ in range(100):
        mujoco.mj_step(phy.m, phy.d, 1)
        viz.viz(phy, False)

    print()



if __name__ == "__main__":
    main()
