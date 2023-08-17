import argparse
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import mujoco.viewer
import numpy as np
import rerun as rr
from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow, QApplication
from scipy.linalg import block_diag

import rospy
from geometry_msgs.msg import Pose, Quaternion
from mjregrasping.basic_3d_pose_marker import Basic3DPoseInteractiveMarker
from mjregrasping.grasping import activate_grasp
from mjregrasping.jacobian_ctrl import get_w_in_tool, warn_near_joint_limits
from mjregrasping.mjsaver import save_data_and_eq, load_data_and_eq
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.my_transforms import xyzw_quat_from_matrix, xyzw_quat_to_matrix
from mjregrasping.physics import Physics, get_q
from mjregrasping.rollout import control_step
from mjregrasping.scenarios import cable_harness, setup_cable_harness
from mjregrasping.viz import make_viz
from ros_numpy import numpify, msgify


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rospy.init_node("viz_demo")
    parser = argparse.ArgumentParser()
    parser.add_argument("demo", type=Path)
    args = parser.parse_args()

    assert args.demo.is_dir()

    scenario = cable_harness
    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))

    rr.init("viewer")
    rr.connect()

    viz = make_viz(scenario)

    paths = sorted(list(args.demo.glob("*.pkl")))
    print(f'Found {len(paths)} paths')
    for path in paths:
        d = load_data_and_eq(m, path, True)
        phy = Physics(m, d, objects=MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))
        viz.viz(phy, is_planning=False)

        time.sleep(phy.m.opt.timestep * 10)


if __name__ == '__main__':
    main()
