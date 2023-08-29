#!/usr/bin/env python3

import mujoco
import numpy as np
from vedo import Line

import rospy
from mjregrasping.homotopy_checker import get_full_h_signature_from_phy
from mjregrasping.homotopy_utils import load_skeletons
from mjregrasping.mjvedo import MjVedo, COLORS
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.physics import Physics
from mjregrasping.scenarios import val_untangle, setup_untangle


def main():
    rospy.init_node("title_fig")

    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    scenario = val_untangle

    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))

    objects = MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
    phy = Physics(m, mujoco.MjData(m), objects)
    mujoco.mj_forward(phy.m, phy.d)
    skeletons = load_skeletons("models/computer_rack_skeleton.hjson")

    setup_untangle(phy, None)
    h, loops = get_full_h_signature_from_phy(skeletons, phy, False, False)

    mjvedo = MjVedo(scenario.xml_path)
    set_cam(mjvedo)
    mjvedo.viz(phy)
    mjvedo.plotter.render().screenshot("title_fig_scene.png", scale=3)

    mjvedo = MjVedo(scenario.xml_path)
    set_cam(mjvedo)
    lw = 25
    mjvedo.viz(phy, is_planning=True)
    for skel in skeletons.values():
        mjvedo.plotter += Line(skel, lw=lw, c='k')
    for i, loop in enumerate(loops):
        mjvedo.plotter += Line(loop, lw=lw, c=COLORS[i % len(COLORS)])
    mjvedo.plotter.render().screenshot("title_fig_skel.png", scale=3)


def set_cam(mjvedo):
    mjvedo.plotter.camera.SetFocalPoint(0, 0.7, 0)
    mjvedo.plotter.camera.SetPosition(2.0, -0.3, 1.8)


if __name__ == "__main__":
    main()
