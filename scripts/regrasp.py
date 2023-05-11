#!/usr/bin/env python3
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import rerun as rr

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.body_with_children import Objects
from mjregrasping.goals import ObjectPointGoal
from mjregrasping.mjsaver import load_data_and_eq
from mjregrasping.movie import MjMovieMaker
from mjregrasping.params import Params
from mjregrasping.regrasp_mpc import RegraspMPC
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rviz import MjRViz
from mjregrasping.viz import Viz

logger = logging.getLogger(f'rosout.{__name__}')


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rr.init('mjregrasping')
    rr.connect()
    rospy.init_node("untangle")
    xml_path = "models/untangle_scene.xml"
    tfw = TF2Wrapper()
    mjviz = MjRViz(xml_path, tfw)
    p = Params()
    viz = Viz(rviz=mjviz, mjrr=MjReRun(xml_path), tfw=tfw, p=p)

    root = Path("results")
    root.mkdir(exist_ok=True)

    for seed in range(p.n_seeds):
        m = mujoco.MjModel.from_xml_path("models/untangle_scene.xml")

        objects = Objects(m)

        d = load_data_and_eq(m)

        mov = MjMovieMaker(m, "rack1")
        mov_path = root / f'untangle_{seed}.mp4'
        mov.start(mov_path, fps=12)

        goal = ObjectPointGoal(model=m,
                               viz=viz,
                               goal_point=np.array([0.78, 0.04, 1.27]),
                               body_idx=-1,
                               goal_radius=0.05,
                               objects=objects)

        with ThreadPoolExecutor(multiprocessing.cpu_count() - 1) as pool:
            mpc = RegraspMPC(m, pool, viz, goal, objects=objects, seed=seed, mov=mov)
            mpc.compute_new_grasp(d)


if __name__ == "__main__":
    main()
