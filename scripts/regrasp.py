#!/usr/bin/env python3
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import rerun as rr

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.body_with_children import Objects
from mjregrasping.dijsktra_field import save_load_dfield
from mjregrasping.goals import ObjectPointGoal
from mjregrasping.grasp_state import GraspState
from mjregrasping.mjsaver import load_data_and_eq
from mjregrasping.params import Params
from mjregrasping.physics import Physics
from mjregrasping.regrasp_mpc import RegraspMPC, vis_regrasp_solutions_and_costs
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rviz import MjRViz
from mjregrasping.viz import Viz

logger = logging.getLogger(f'rosout.{__name__}')


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rr.init('mjregrasping')
    rr.connect()
    rospy.init_node("regrasp")
    xml_path = "models/untangle_scene.xml"
    tfw = TF2Wrapper()
    mjviz = MjRViz(xml_path, tfw)
    p = Params()
    viz = Viz(rviz=mjviz, mjrr=MjReRun(xml_path), tfw=tfw, p=p)

    seed = 1
    m = mujoco.MjModel.from_xml_path("models/untangle_scene.xml")

    objects = Objects(m)

    d = load_data_and_eq(m)
    phy = Physics(m, d)
    goal_point = np.array([0.78, 0.04, 1.27])
    dfield = save_load_dfield(phy, goal_point)

    for _ in range(5):
        mjviz.viz(phy, is_planning=False)

    goal = ObjectPointGoal(dfield=dfield,
                           viz=viz,
                           goal_point=np.array([0.78, 0.04, 1.27]),
                           body_idx=-1,
                           goal_radius=0.05,
                           objects=objects)

    with ThreadPoolExecutor(multiprocessing.cpu_count() - 1) as pool:
        mpc = RegraspMPC(mppi_nu=m.nu, pool=pool, viz=viz, goal=goal, objects=objects, seed=seed, mov=None)

        # Debug why [0.0, 0.59] is better than [0.73, 0] for the first re-grasp???
        grasp_locations1 = np.array([0.73, 0])
        is_grasping = np.array([1, 0])
        grasp0 = GraspState.from_mujoco(mpc.rope_body_indices, phy.m)
        costs1 = mpc.score_grasp_location(grasp_locations1, grasp0, is_grasping, phy)
        grasp_locations2 = np.array([0, 0.59])
        is_grasping = np.array([0, 1])
        costs2 = mpc.score_grasp_location(grasp_locations2, grasp0, is_grasping, phy)
        costs_lists = [costs1, costs2]
        costs = [sum(costs_i) for costs_i in costs_lists]
        vis_regrasp_solutions_and_costs(is_grasping, costs_lists, [grasp_locations1, grasp_locations2])
        print(costs)

        print("done")

        mpc.compute_new_grasp(phy)


if __name__ == "__main__":
    main()
