#!/usr/bin/env python3
import logging
from time import perf_counter

import mujoco
import numpy as np
import rerun as rr

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.ik import position_jacobian
from mjregrasping.params import Params
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rviz import MjRViz
from mjregrasping.viz import Viz


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rr.init('test_inverse_control')
    rr.connect()
    rospy.init_node("test_inverse_control")
    xml_path = "models/pull_scene.xml"
    tfw = TF2Wrapper()
    mjviz = MjRViz(xml_path, tfw)
    p = Params()
    viz = Viz(rviz=mjviz, mjrr=MjReRun(xml_path), tfw=tfw, p=p)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    phy = Physics(m, d)
    mujoco.mj_forward(phy.m, phy.d)

    body_idx = phy.m.body("drive10").id
    target_point = np.array([1.046, 0.13, 0.1])
    current_target_point = target_point
    rng = np.random.RandomState(0)
    t0 = perf_counter()

    for i in range(1000):
        ctrl, position_error = position_jacobian(phy, body_idx, current_target_point)

        # if i % 100 == 0 or position_error < 0.005:
        #     print(f'dt: {perf_counter() - t0:.4f}')
        #     t0 = perf_counter()
        #     current_target_point = target_point + rng.uniform(-0.3, 0.3, size=3)

        np.copyto(phy.d.ctrl, ctrl)
        mujoco.mj_step(phy.m, phy.d, nstep=25)

        viz.viz(phy)
        viz.sphere(ns='goal', position=current_target_point, radius=0.01, color=(0, 1, 0, 1), frame_id='world', idx=0)
        rospy.sleep(0.002)


if __name__ == "__main__":
    main()
