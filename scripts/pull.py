#!/usr/bin/env python3

import numpy as np

from arc_utilities import ros_init
from mjregrasping.run_evaluation import run_evaluation
from mjregrasping.scenarios import val_pull, setup_pull_scene, make_pull_goal


@ros_init.with_ros("pull")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    run_evaluation(
        scenario=val_pull,
        make_goal=make_pull_goal,
        skeletons={},
        setup_scene=setup_pull_scene,
        seeds=[2]
    )


if __name__ == "__main__":
    main()
