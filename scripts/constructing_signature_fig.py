#!/usr/bin/env python3
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
from vedo import Line

import rospy
from mjregrasping.grasping import activate_grasp
from mjregrasping.homotopy_checker import get_full_h_signature_from_phy
from mjregrasping.mjvedo import MjVedo, COLORS
from mjregrasping.scenarios import threading_cable
from mjregrasping.trials import load_trial


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    scenario = threading_cable

    gl_ctx = mujoco.GLContext(1280, 720)
    gl_ctx.make_current()

    qpos_filename = Path('results/Threading/1694678882_0/Threading_1694678882_0_\signature{}_qpos.npy')
    outdir = qpos_filename.parent
    qpos = np.load(qpos_filename)

    trial_idx = int(qpos_filename.stem.split('_')[2])
    phy, sdf, skeletons, mov = load_trial(trial_idx, gl_ctx, scenario, viz=None)

    # Load the given frame and render it with mjvedo
    frame_idx = 8000
    phy.d.qpos = qpos[frame_idx]
    mujoco.mj_forward(phy.m, phy.d)

    activate_grasp(phy, 'left', 1)
    activate_grasp(phy, 'right', 0.94)

    t0 = perf_counter()
    h, loops = get_full_h_signature_from_phy(skeletons, phy, False, False)
    print(f'get_full_h_signature_from_phy took {perf_counter() - t0:.3f}s')

    mjvedo = MjVedo(scenario.xml_path)
    set_cam(mjvedo)
    mjvedo.viz(phy)
    mjvedo.plotter.render().screenshot(outdir / f"scene_{trial_idx}_{frame_idx}.png", scale=3)

    mjvedo = MjVedo(scenario.xml_path)
    set_cam(mjvedo)
    lw = 25
    mjvedo.viz(phy, is_planning=True)
    for skel in skeletons.values():
        mjvedo.plotter += Line(skel, lw=lw, c='k', alpha=0.8)
    for i, loop in enumerate(loops):
        loop_viz = loop + i * 0.002  # add noise to avoid z-fighting and overlapping lines
        mjvedo.plotter += Line(loop_viz, lw=lw, c=COLORS[i % len(COLORS)])
    mjvedo.plotter.render().screenshot(outdir / f"skel_{trial_idx}_{frame_idx}.png", scale=3)


def set_cam(mjvedo):
    mjvedo.plotter.camera.SetFocalPoint(0, 0.7, -0.1)
    mjvedo.plotter.camera.SetPosition(2.0, -0.7, 2.4)
    mjvedo.plotter.camera.SetViewUp(0, 0, 1)


if __name__ == "__main__":
    main()
