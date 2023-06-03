#!/usr/bin/env python3

import argparse

import mujoco
import numpy as np

from mjregrasping.initialize import initialize
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.physics import Physics
from mjregrasping.rollout import DEFAULT_SUB_TIME_S


def main():
    np.set_printoptions(precision=4, suppress=True)

    m, d, viz = initialize("val_inverse_control", "models/val_husky.xml")
    phy = Physics(m, d)

    for i in range(100):
        # Compute the control using inverse dynamics
        phy.d.
        mujoco.mj_inverse(phy.m, phy.d)
        phy.d.qfrc_inverse
        viz.viz(phy)


if __name__ == "__main__":
    main()
