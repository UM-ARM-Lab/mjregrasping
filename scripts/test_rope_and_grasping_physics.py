#!/usr/bin/env python3
import logging
from pathlib import Path

import mujoco
import numpy as np
import rerun as rr

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.body_with_children import Objects
from mjregrasping.grasping import deactivate_eq, activate_eq
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.params import Params
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.rviz import MjRViz
from mjregrasping.scenes import setup_tangled_scene, settle
from mjregrasping.viz import Viz

logger = logging.getLogger(f'rosout.{__name__}')


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rr.init('mjregrasping')
    rr.connect()
    rospy.init_node("untangle")
    xml_path = "models/untangle_scene.xml"
    tfw = TF2Wrapper()
    mjviz = MjRViz(xml_path, tfw)
    p = Params()
    viz = Viz(rviz=mjviz, mjrr=MjReRun(xml_path), tfw=tfw, p=p)

    root = Path("results")
    root.mkdir(exist_ok=True)

    m = mujoco.MjModel.from_xml_path("models/untangle_scene.xml")
    objects = Objects(m)
    d = mujoco.MjData(m)
    phy = Physics(m, d)

    setup_tangled_scene(phy, viz)

    for _ in range(3):
        robot_q1 = np.array([
            -0.7, -0.8,  # torso
            -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
            0, 0,  # left gripper
            -0.9, -0.2, 0, 0.30, 0, -0.2, 0,  # right arm
            0, 0,  # right gripper
        ])
        pid_to_joint_config(phy, viz, robot_q1, sub_time_s=DEFAULT_SUB_TIME_S)
        robot_q1 = np.array([
            -0.7, 0.8,  # torso
            -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
            0, 0,  # left gripper
            -0.9, -0.2, 0, 0.30, 0, -0.2, 0,  # right arm
            0, 0,  # right gripper
        ])
        pid_to_joint_config(phy, viz, robot_q1, sub_time_s=DEFAULT_SUB_TIME_S)

    activate_eq(phy.m, 'left')
    deactivate_eq(phy.m, 'right')
    settle(phy, DEFAULT_SUB_TIME_S, viz, is_planning=False, settle_steps=100)
    deactivate_eq(phy.m, 'left')
    settle(phy, DEFAULT_SUB_TIME_S, viz, is_planning=False, settle_steps=100)

    pid_to_joint_config(phy, viz, robot_q1, sub_time_s=DEFAULT_SUB_TIME_S)


if __name__ == "__main__":
    main()
