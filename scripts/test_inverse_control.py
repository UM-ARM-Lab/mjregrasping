#!/usr/bin/env python3
import logging

import mujoco
import numpy as np
import rerun as rr

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.params import Params
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rviz import MjRViz
from mjregrasping.viz import Viz

logger = logging.getLogger(f'rosout.{__name__}')


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rr.init('test_inverse_control')
    rr.connect()
    rospy.init_node("test_inverse_control")
    xml_path = "models/val_husky.xml"
    tfw = TF2Wrapper()
    mjviz = MjRViz(xml_path, tfw)
    p = Params()
    viz = Viz(rviz=mjviz, mjrr=MjReRun(xml_path), tfw=tfw, p=p)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    phy = Physics(m, d)
    mujoco.mj_forward(phy.m, phy.d)

    for i in range(350):
        # compute the "end-effector" jacobian which relates the
        # joint velocities to the end-effector velocity
        # where the ee_offset specifies the location/offset from the end-effector body
        ee_offset = np.zeros(3)
        body_idx = phy.m.body("drive50").id
        Jp = np.zeros((3, phy.m.nv))
        Jr = np.zeros((3, phy.m.nv))
        mujoco.mj_jac(m, d, Jp, Jr, ee_offset, body_idx)
        J = np.concatenate((Jp, -Jr), axis=0)
        J_T = J.T

        target_point = np.array([1.0, 0.0, 0.5])
        # TODO: account for ee_offset here
        current_ee_pos = phy.d.body("drive50").xpos
        ee_vel_p = target_point - current_ee_pos  # Order of subtraction matters here!
        ee_vel = np.concatenate((ee_vel_p, np.zeros(3)), axis=0)

        # print(ee_vel)
        eps = 1e-3
        ctrl = J_T @ np.linalg.solve(J @ J_T + eps * np.eye(6), ee_vel)
        # rescale to respect velocity limits
        # TODO: use nullspace to respect joint limits by trying to move towards the home configuration
        vmin = phy.m.actuator_ctrlrange[:, 0]
        vmax = phy.m.actuator_ctrlrange[:, 1]

        if np.any(ctrl > vmax):
            offending_joint = np.argmax(ctrl)
            ctrl = ctrl / np.max(ctrl) * vmax[offending_joint]
        elif np.any(ctrl < vmin):
            offending_joint = np.argmin(ctrl)
            ctrl = ctrl / np.min(ctrl) * vmin[offending_joint]

        np.copyto(phy.d.ctrl, ctrl)
        mujoco.mj_step(phy.m, phy.d, nstep=25)
        viz.viz(phy)
        rospy.sleep(0.002)


if __name__ == "__main__":
    main()
