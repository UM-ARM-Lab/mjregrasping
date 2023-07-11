#!/usr/bin/env python3
import time

import mujoco
import numpy as np
import rerun as rr

from arc_utilities import ros_init
from mjregrasping.magnetic_fields import load_skeletons
from mjregrasping.mujoco_objects import Objects
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import log_skeletons
from mjregrasping.scenarios import cable_harness, setup_cable_harness
from mjregrasping.viz import make_viz


@ros_init.with_ros("cable_harness")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('regrasp_mpc_runner')
    rr.connect()

    scenario = cable_harness

    viz = make_viz(scenario)

    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
    d = mujoco.MjData(m)
    objects = Objects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
    phy = Physics(m, d, objects)

    setup_cable_harness(phy, viz)

    skeletons = load_skeletons(scenario.skeletons_path)
    log_skeletons(skeletons)

    viz.viz(phy)

    # run_evaluation(
    #     scenario=val_untangle,
    #     make_goal=make_untangle_goal,
    #     skeletons=load_skeletons(val_untangle.skeletons_path),
    #     setup_scene=setup_untangle,
    #     seeds=[1],
    # )


if __name__ == "__main__":
    main()
