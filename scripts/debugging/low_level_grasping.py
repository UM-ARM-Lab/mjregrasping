from pathlib import Path
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import open3d as o3d
import pymanopt
import pyrealsense2 as rs
import rerun as rr
import torch
from arm_segmentation.predictor import Predictor
from pymanopt.manifolds import SpecialOrthogonalGroup

import ros_numpy
import rospy
from arc_utilities import ros_init
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.homotopy_utils import make_ring_skeleton, skeleton_field_dir
from mjregrasping.jacobian_ctrl import get_jacobian_ctrl
from mjregrasping.low_level_grasping import run_grasp_controller
from mjregrasping.movie import MjRGBD
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.my_transforms import mj_transform_points, np_wxyz_to_xyzw
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.real_val import RealValCommander
from mjregrasping.rviz import plot_points_rviz
from mjregrasping.scenarios import val_untangle
from mjregrasping.viz import make_viz, Viz
from ros_numpy.point_cloud2 import merge_rgb_fields
from sensor_msgs.msg import PointCloud2, Image


def batch_rotate_and_translate(points, mat, pos=None):
    new_p = (mat @ points.T).T
    if pos is not None:
        new_p += pos
    return new_p


@ros_init.with_ros("low_level_grasping")
def main():
    np.set_printoptions(precision=5, suppress=True, linewidth=200)
    scenario = val_untangle
    scenario.xml_path = Path("models/real_scene.xml")
    scenario.obstacle_name = "obstacles"

    rr.init('low_level_grasping')
    rr.connect()

    viz: Viz = make_viz(scenario)

    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
    objects = MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)

    d = mujoco.MjData(m)
    phy = Physics(m, d, objects)

    val_cmd = RealValCommander(phy)

    run_grasp_controller(val_cmd, phy, tool_idx=1, viz=viz, finger_q_open=0.5, finger_q_closed=0.05)


if __name__ == '__main__':
    main()
