#!/usr/bin/env python3
from itertools import cycle
from pathlib import Path

import numpy as np
from vedo import Line, Text2D

from mjregrasping.homotopy_checker import get_loops_from_phy
from mjregrasping.mjvedo import MjVedo, COLORS, set_phy_to_frame, load_from_npy
from mjregrasping.scenarios import val_untangle


def ease_out(t):
    """ input and output are from 0 to 1 """
    return np.clip(np.cos(t * np.pi / 2), 0, 1)


def ease_out_in(t, c=1.0):
    """ input and output are from 0 to 1 """
    return np.clip(1 + (np.cos(2 * t * np.pi) - 1), 0, 1)


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    qpos_path = Path(
        "/home/peter/mjregrasping_ws/src/mjregrasping/results/Untangle/untangle_for_animiation/1695162530_11/Untangle_1695162530_11_\signature{}_qpos.npy")
    outdir, phy, qpos, skeletons, trial_idx = load_from_npy(qpos_path, val_untangle)

    mjvedo = MjVedo(val_untangle.xml_path)
    mjvedo.plotter += Text2D(f"Untangling A Cable", 'bottom-center')

    cx = 0
    cy = 0.5
    cz = 0.1
    distance = 2.5
    start_azimuth = 0
    num_frames = 3000
    rot_per_frame = np.deg2rad(-0.25)
    z = 1.5
    dz_per_frame = 0.0005
    sim_steps_per_frame = 3

    mjvedo.viz(phy)
    fade_out_actors = mjvedo.get_object_actors(phy.o.obstacle)
    fade_out_actors += mjvedo.get_object_actors(phy.o.robot)
    fade_out_actors += mjvedo.get_object_actors(phy.o.rope)
    fade_out_actors.remove(mjvedo.actor_map[phy.m.geom("goal").id])

    mjvedo.plotter.camera.SetViewUp(0, 0, 1)

    def sec2frame(s: float):
        return int(s * mjvedo.fps)

    qpos_t = 0.  # float here so we don't lose precision
    azimuth = start_azimuth
    lines = []

    def anim(t, plotter):
        nonlocal qpos_t, lines, azimuth, z, cz

        set_phy_to_frame(phy, qpos, int(qpos_t))
        mjvedo.viz(phy)

        start_spin = sec2frame(3)
        stop_spin = sec2frame(7)
        if t == start_spin:
            _, loops = get_loops_from_phy(phy)
            for loop, c in zip(loops, cycle(COLORS)):
                loop_line = Line(loop, lw=10, alpha=1, c=c)
                mjvedo.plotter += loop_line
                lines.append(loop_line)
            for name, skel in skeletons.items():
                skel_line = Line(skel, lw=10, alpha=1)
                mjvedo.plotter += skel_line
                lines.append(skel_line)
        elif start_spin < t < stop_spin:
            frac_spin = (t - start_spin) / (stop_spin - start_spin)
            alpha = ease_out_in(frac_spin, c=1.5)
            for a in fade_out_actors:
                a.alpha(alpha)

            azimuth += rot_per_frame
        elif t == stop_spin:
            # remove the lines
            for line in lines:
                mjvedo.plotter -= line
            for a in fade_out_actors:
                a.alpha(1)
        else:
            qpos_t += sim_steps_per_frame
            if qpos_t >= len(qpos):
                return True

        x = distance * np.cos(azimuth) + cx
        y = distance * np.sin(azimuth) + cy
        z += dz_per_frame
        mjvedo.plotter.camera.SetFocalPoint(cx, cy, cz)
        plotter.camera.SetPosition(x, y, z)

        return False

    mjvedo.record(f"results/untangle_example_anim_{trial_idx}.mp4", num_frames=num_frames, anim_func=anim)


if __name__ == "__main__":
    main()
