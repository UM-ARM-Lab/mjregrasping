#!/usr/bin/env python3
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
from vedo import Line, DashedLine

from mjregrasping.grasping import activate_grasp
from mjregrasping.homotopy_checker import get_full_h_signature_from_phy
from mjregrasping.mjvedo import MjVedo, COLORS, load_frame_from_npy
from mjregrasping.scenarios import threading_cable


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    scenario = threading_cable

    gl_ctx = mujoco.GLContext(1280, 720)
    gl_ctx.make_current()

    qpos_filename = Path('results/constructing_signature_threading_0_ours_qpos.npy')
    frame_idx = 8000
    outdir, phy, qpos, skeletons, trial_idx = load_frame_from_npy(frame_idx, qpos_filename, scenario)

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
        mjvedo.plotter += DashedLine(loop_viz, lw=lw, c=COLORS[i % len(COLORS)], spacing=0.3)
    mjvedo.plotter.render().screenshot(outdir / f"skel_{trial_idx}_{frame_idx}.png", scale=3)


def set_cam(mjvedo):
    mjvedo.plotter.camera.SetFocalPoint(0, 0.7, -0.1)
    mjvedo.plotter.camera.SetPosition(2.0, -0.7, 2.4)
    mjvedo.plotter.camera.SetViewUp(0, 0, 1)


if __name__ == "__main__":
    main()
