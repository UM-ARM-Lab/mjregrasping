#!/usr/bin/env python3
import logging
import multiprocessing
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import rerun as rr

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.body_with_children import Objects
from mjregrasping.dijsktra_field import make_dfield
from mjregrasping.goals import ObjectPointGoal
from mjregrasping.movie import MjMovieMaker
from mjregrasping.params import Params
from mjregrasping.physics import Physics
from mjregrasping.regrasp_mpc import RegraspMPC
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rviz import MjRViz
from mjregrasping.scenes import setup_tangled_scene
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

    goal_point = np.array([0.73, 0.04, 1.25])
    n_seeds = 5
    for seed in range(n_seeds):
        m = mujoco.MjModel.from_xml_path("models/untangle_scene.xml")
        objects = Objects(m)
        d = mujoco.MjData(m)
        phy = Physics(m, d)

        # setup_untangled_scene(phy, mjviz)
        setup_tangled_scene(phy, viz)

        mov = MjMovieMaker(m, "rack1")
        mov_path = root / f'untangle_{seed}.mp4'
        mov.start(mov_path, fps=12)

        # store and load from disk to save time?
        dfield = save_load_dfield(phy, goal_point)

        goal = ObjectPointGoal(dfield=dfield,
                               viz=viz,
                               goal_point=goal_point,
                               body_idx=-1,
                               goal_radius=0.05,
                               objects=objects)

        with ThreadPoolExecutor(multiprocessing.cpu_count() - 1) as pool:
            mpc = RegraspMPC(phy.m, pool, viz, goal, objects=objects, seed=seed, mov=mov)
            result = mpc.run(phy)
            logger.info(f"{seed=} {result=}")


def save_load_dfield(phy, goal_point):
    dfield_path = Path("models/dfield.pkl")
    if dfield_path.exists():
        with dfield_path.open('rb') as f:
            dfield = pickle.load(f)
    else:
        res = 0.02
        extents_2d = np.array([[0.6, 1.4], [-0.7, 0.4], [0.2, 1.3]])
        dfield = make_dfield(phy, extents_2d, res, goal_point)
        with dfield_path.open('wb') as f:
            pickle.dump(dfield, f)
    return dfield


if __name__ == "__main__":
    main()
