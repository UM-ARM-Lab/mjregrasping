#!/usr/bin/env python3
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
import pysdf_tools
import rerun as rr
from PIL import Image

from arc_utilities import ros_init
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.grasping import get_grasp_locs
from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner, get_geodesic_dist
from mjregrasping.homotopy_utils import load_skeletons
from mjregrasping.mjsaver import load_data_and_eq
from mjregrasping.movie import MjRenderer
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.params import Params
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rerun_visualizer import log_skeletons
from mjregrasping.rrt import GraspRRT
from mjregrasping.rviz import MjRViz
from mjregrasping.scenarios import val_untangle, make_untangle_goal, cable_harness
from mjregrasping.sdf_collision_checker import SDFCollisionChecker
from mjregrasping.viz import Viz


@ros_init.with_ros("test_regrasp_homotopy")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rr.init('test_regrasp_homotopy')
    rr.connect()

    scenario = cable_harness
    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))

    tfw = TF2Wrapper()
    mjviz = MjRViz(scenario.xml_path, tfw)
    p = Params()

    viz = Viz(rviz=mjviz, mjrr=MjReRun(scenario.xml_path), tfw=tfw, p=p)
    phy = Physics(m, mujoco.MjData(m),
                  objects=MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))
    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)

    skeletons = load_skeletons(scenario.skeletons_path)
    sdf = pysdf_tools.SignedDistanceField.LoadFromFile(str(scenario.sdf_path))
    cc = SDFCollisionChecker(sdf)

    goal = make_untangle_goal(viz)
    r = MjRenderer(m)

    rr.set_time_sequence('homotopy', 0)
    states_paths = [
        Path("states/CableHarness/stuck1.pkl"),
        # Path("states/on_stuck/1690927082.pkl"),
        # Path("states/Untangle/1690834987.pkl"),
    ]
    grasp_rrt = GraspRRT()
    for state_path in states_paths:
        for seed in range(1, 5):
            planner = HomotopyRegraspPlanner(goal, grasp_rrt, skeletons, seed=seed)
            d = load_data_and_eq(m, state_path, True)
            phy = Physics(m, d, objects=MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))

            viz.viz(phy)
            rr.log_cleared("planned", recursive=True)
            log_skeletons(skeletons, color=(0, 255, 0, 255), timeless=True)

            img = r.render(d)
            img_path = (state_path.parent / state_path.stem).with_suffix(".png")
            Image.fromarray(img).save(img_path)

            initial_geodesic_cost = get_geodesic_dist(get_grasp_locs(phy), goal)

            grasps_inputs = planner.sample_grasp_inputs(phy)
            # t0 = perf_counter()
            # sim_grasps = planner.simulate_grasps_parallel(grasps_inputs, phy)
            # t1 = perf_counter()
            # print(f'parallel: {t1 - t0:.3f}s')
            t0 = perf_counter()
            sim_grasps = planner.simulate_grasps(grasps_inputs, phy, viz=None, viz_execution=True)
            t1 = perf_counter()
            print(f'planning time: {t1 - t0:.3f}s')
            best_grasp = planner.get_best(sim_grasps, viz=None)

            costs = [planner.cost(g, None) for g in sim_grasps]
            print(f'grasp planning: {t1 - t0:.3f}s {best_grasp.locs=:}')

            # visualize the grasps in order of cost
            rr.set_time_sequence('homotopy', 0)
            sorted_i = np.argsort(costs)[::-1]
            t = 0
            for i in sorted_i:
                rr.set_time_sequence('homotopy', t)
                t += 1
                grasp = sim_grasps[i]
                cost_i = costs[i]
                geodesic_cost_i = get_geodesic_dist(get_grasp_locs(grasp.phy), goal)
                print(f'cost: {cost_i:.3f} {grasp.locs=:} {grasp.strategy=:}, geodesic: {geodesic_cost_i:.3f}')
                rr.log_text_entry("planned", f'cost: {cost_i:.3f} grasp: {grasp.locs=:}')
                viz.viz(grasp.phy, is_planning=True)
            print(f'{initial_geodesic_cost=:.3f}')

            # Update blacklist and recompute cost
            planner.update_blacklists(phy)
            best_grasp = planner.get_best(sim_grasps, viz=None)
            costs = [planner.cost(g, None) for g in sim_grasps]
            print(f'after blacklisting: {best_grasp.locs=:}')

            sorted_i = np.argsort(costs)[::-1]
            for i in sorted_i:
                rr.set_time_sequence('homotopy', t)
                t += 1
                grasp = sim_grasps[i]
                cost_i = costs[i]
                geodesic_cost_i = get_geodesic_dist(get_grasp_locs(grasp.phy), goal)
                print(f'cost: {cost_i:.3f} {grasp.locs=:} {grasp.strategy=:}, geodesic: {geodesic_cost_i:.3f}')
                rr.log_text_entry("planned", f'cost: {cost_i:.3f} grasp: {grasp.locs=:}')
                viz.viz(grasp.phy, is_planning=True)
            print(f'{initial_geodesic_cost=:.3f}')


if __name__ == "__main__":
    main()
