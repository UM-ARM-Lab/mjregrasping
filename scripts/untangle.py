#!/usr/bin/env python3
import numpy as np

from arc_utilities import ros_init
from mjregrasping.homotopy_utils import load_skeletons
from mjregrasping.run_evaluation import run_evaluation
from mjregrasping.scenarios import val_untangle, make_untangle_goal, setup_untangle


@ros_init.with_ros("untangle")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    run_evaluation(
        scenario=val_untangle,
        make_goal=make_untangle_goal,
        skeletons=load_skeletons(val_untangle.skeletons_path),
        setup_scene=setup_untangle,
        seeds=[2],
    )


if __name__ == "__main__":
    main()
