#!/usr/bin/env python3
import numpy as np

from arc_utilities import ros_init
from mjregrasping.magnetic_fields import load_skeletons
from mjregrasping.run_evaluation import run_evaluation
from mjregrasping.scenarios import conq_hose, setup_conq_hose_scene, make_conq_hose_goal


@ros_init.with_ros("conq_hose")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    run_evaluation(
        scenario=conq_hose,
        make_goal=make_conq_hose_goal,
        skeletons=load_skeletons(conq_hose.skeletons_path),
        setup_scene=setup_conq_hose_scene,
        seeds=[1]
    )


if __name__ == "__main__":
    main()
