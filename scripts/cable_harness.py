#!/usr/bin/env python3
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import pysdf_tools
import rerun as rr
from colorama import Fore

from arc_utilities import ros_init
from mjregrasping.homotopy_utils import load_skeletons
from mjregrasping.movie import MjMovieMaker
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.physics import Physics
from mjregrasping.regrasp_mpc import RegraspMPC, UnsolvableException
from mjregrasping.scenarios import cable_harness, setup_cable_harness, make_ch_goal1, make_ch_goal2
from mjregrasping.viz import make_viz


@ros_init.with_ros("cable_harness")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('regrasp_mpc_runner')
    rr.connect()

    scenario = cable_harness

    viz = make_viz(scenario)

    root = Path("results") / scenario.name
    root.mkdir(exist_ok=True, parents=True)

    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
    d = mujoco.MjData(m)
    objects = MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
    phy = Physics(m, d, objects)

    sdf = pysdf_tools.SignedDistanceField.LoadFromFile(str(scenario.sdf_path))
    # viz.sdf(sdf, '', 0)
    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)

    setup_cable_harness(phy, viz)

    mov = MjMovieMaker(m)
    now = int(time.time())
    seed = 0
    mov_path = root / f'seed_{seed}_{now}.mp4'
    print(f"Saving movie to {mov_path}")
    mov.start(mov_path, fps=8)


    with ThreadPoolExecutor(multiprocessing.cpu_count() - 1) as pool:
        mpc = RegraspMPC(pool, phy.m.nu, load_skeletons(scenario.skeletons_path), sdf, seed, viz, mov)
        goal1 = make_ch_goal1(viz)
        goal2 = make_ch_goal2(viz)
        mpc.run_threading_goal(phy, [goal1, goal2])
        mpc.close()


if __name__ == "__main__":
    main()
