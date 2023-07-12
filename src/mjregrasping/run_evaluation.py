import multiprocessing
import time
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import mujoco
import pysdf_tools
import rerun as rr

from mjregrasping.movie import MjMovieMaker
from mjregrasping.mujoco_objects import Objects
from mjregrasping.physics import Physics
from mjregrasping.regrasp_mpc import RegraspMPC
from mjregrasping.scenarios import Scenario
from mjregrasping.viz import make_viz


def run_evaluation(scenario: Scenario, skeletons, make_goal: Callable, setup_scene: Callable, seeds):
    rr.init('regrasp_mpc_runner')
    rr.connect()

    viz = make_viz(scenario)

    root = Path("results") / scenario.name
    root.mkdir(exist_ok=True, parents=True)

    for seed in seeds:
        m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
        d = mujoco.MjData(m)
        objects = Objects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
        phy = Physics(m, d, objects)

        sdf = pysdf_tools.SignedDistanceField.LoadFromFile(str(scenario.sdf_path))
        # viz.sdf(sdf, '', 0)
        mujoco.mj_forward(phy.m, phy.d)
        viz.viz(phy)

        setup_scene(phy, viz)

        mov = MjMovieMaker(m)
        now = int(time.time())
        mov_path = root / f'seed_{seed}_{now}.mp4'
        print(f"Saving movie to {mov_path}")
        mov.start(mov_path, fps=12)

        goal = make_goal(viz)

        with ThreadPoolExecutor(multiprocessing.cpu_count() - 1) as pool:
            mpc = RegraspMPC(pool, phy.m.nu, skeletons, sdf, goal, seed, viz, mov)
            mpc.run(phy)
            mpc.close()
