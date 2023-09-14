#!/usr/bin/env python3
from time import perf_counter

import mujoco
import numpy as np
from vedo import Line

import rospy
from mjregrasping.homotopy_checker import get_full_h_signature_from_phy
from mjregrasping.mjvedo import MjVedo, COLORS
from mjregrasping.scenarios import val_untangle
from mjregrasping.trials import load_trial, load_phy_and_skeletons
from mjregrasping.viz import make_viz


def main():
    rospy.init_node("title_fig")

    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    scenario = val_untangle

    viz = make_viz(scenario)

    for trial_idx in range(25):
        phy, _, skeletons = load_phy_and_skeletons(trial_idx, scenario)

        t0 = perf_counter()
        h, loops = get_full_h_signature_from_phy(skeletons, phy, False, False)
        print(f'get_full_h_signature_from_phy took {perf_counter() - t0:.3f}s')

        mjvedo = MjVedo(scenario.xml_path)
        set_cam(mjvedo)
        mjvedo.viz(phy)
        mjvedo.plotter.render().screenshot(f"title_fig_scene_{trial_idx}.png", scale=3)

        mjvedo = MjVedo(scenario.xml_path)
        set_cam(mjvedo)
        lw = 25
        mjvedo.viz(phy, is_planning=True)
        for skel in skeletons.values():
            mjvedo.plotter += Line(skel, lw=lw, c='k')
        for i, loop in enumerate(loops):
            mjvedo.plotter += Line(loop, lw=lw, c=COLORS[i % len(COLORS)])
        mjvedo.plotter.render().screenshot(f"title_fig_skel_{trial_idx}.png", scale=3)


def set_cam(mjvedo):
    mjvedo.plotter.camera.SetFocalPoint(0, 0.7, 0)
    mjvedo.plotter.camera.SetPosition(2.0, -0.3, 1.8)
    mjvedo.plotter.camera.SetViewUp(0, 0, 1)


if __name__ == "__main__":
    main()
