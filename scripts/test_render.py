#!/usr/bin/env python3

import argparse
import mujoco
import numpy as np

from mjregrasping.initialize import initialize
from mjregrasping.movie import MjMovieMaker


def main():
    np.set_printoptions(precision=4, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("xml_path", type=str)

    args = parser.parse_args()

    m, d, viz = initialize("test_render", args.xml_path)

    mov = MjMovieMaker(m, "rack1", w=1280, h=720)
    mov.cam.distance = 3.0
    mov.cam.azimuth = 45
    nstep = 20
    mov.start('test.mp4', fps=int(1 / (m.opt.timestep * nstep)))

    for t in range(100):
        mujoco.mj_step(m, d, nstep=nstep)
        mov.render(d)

    mov.close()


if __name__ == "__main__":
    main()
