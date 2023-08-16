#!/usr/bin/env python3

import numpy as np
import rerun as rr

from arc_utilities import ros_init
from mjregrasping.homotopy_utils import load_skeletons
from mjregrasping.run_evaluation import run_evaluation
from mjregrasping.scenarios import cable_harness, setup_cable_harness, make_ch_goal1


@ros_init.with_ros("cable_harness")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    run_evaluation(
        scenario=cable_harness,
        make_goal=make_ch_goal1,
        skeletons=load_skeletons(cable_harness.skeletons_path),
        setup_scene=setup_cable_harness,
        seeds=[1],
    )


if __name__ == "__main__":
    main()
