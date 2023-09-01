#!/usr/bin/env python3
from time import perf_counter

import mujoco
import numpy as np
import rerun as rr

from arc_utilities import ros_init
from mjregrasping.goals import point_goal_from_geom, GraspLocsGoal
from mjregrasping.grasping import get_grasp_locs
from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner
from mjregrasping.regrasp_planner_utils import get_geodesic_dist
from mjregrasping.rollout import control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import threading
from mjregrasping.trials import load_trial
from mjregrasping.viz import make_viz


@ros_init.with_ros("test_regrasp_homotopy")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rr.init('test_regrasp_homotopy')
    rr.connect()

    scenario = threading

    viz = make_viz(scenario)

    gl_ctx = mujoco.GLContext(1280, 720)
    gl_ctx.make_current()

    i = 1
    phy, sdf, skeletons, mov = load_trial(i, gl_ctx, scenario, viz)
    grasp_rrt = GraspRRT()

    grasp_goal = GraspLocsGoal(get_grasp_locs(phy))
    goal = point_goal_from_geom(grasp_goal, phy, "goal", 1, viz)
    planner = HomotopyRegraspPlanner(goal.loc, grasp_rrt, skeletons, seed=i)

    initial_geodesic_cost = get_geodesic_dist(get_grasp_locs(phy), goal.loc)

    ctrl = np.zeros(phy.m.nu)
    ctrl[0] = 0.02
    ctrl[1] = 0.05
    ctrl[10] = 0.05
    ctrl[15] = -0.1
    ctrl[16] = 0.1
    for _ in range(23):
        control_step(phy, ctrl, 0.1)
        viz.viz(phy)
    ctrl = np.zeros(phy.m.nu)
    ctrl[16] = 0.4
    for _ in range(10):
        control_step(phy, ctrl, 0.1)
        viz.viz(phy)

    # from mjregrasping.grasp_strategies import Strategies
    # strategy = [Strategies.NEW_GRASP, Strategies.STAY]
    # candidate_locs = np.array([0.99, 0.93])
    #
    # print('is valid?', grasp_rrt.is_state_valid(phy))
    # res, scene_msg = grasp_rrt.plan(phy, strategy, candidate_locs, viz)
    # print(res.error_code.val)
    # grasp_rrt.display_result(viz, res, scene_msg)

    grasps_inputs = planner.sample_grasp_inputs(phy)
    t0 = perf_counter()
    sim_grasps = planner.simulate_grasps(grasps_inputs, phy, viz=viz, viz_execution=True)
    t1 = perf_counter()
    print(f'planning time: {t1 - t0:.3f}s')
    best_grasp = planner.get_best(sim_grasps, viz=None)

    costs = [planner.costs(g) for g in sim_grasps]
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
        geodesic_cost_i = get_geodesic_dist(get_grasp_locs(grasp.phy), goal.loc)
        print(f'cost: {cost_i:.3f} {grasp.locs=:} {grasp.strategy=:}, geodesic: {geodesic_cost_i:.3f}')
        rr.log_text_entry("planned", f'cost: {cost_i:.3f} grasp: {grasp.locs=:}')
        viz.viz(grasp.phy, is_planning=True)
    print(f'{initial_geodesic_cost=:.3f}')

    # Update blacklist and recompute cost
    planner.update_blacklists(phy)
    best_grasp = planner.get_best(sim_grasps, viz=None)
    costs = [planner.cost(g) for g in sim_grasps]
    print(f'after blacklisting: {best_grasp.locs=:}')

    sorted_i = np.argsort(costs)[::-1]
    for i in sorted_i:
        rr.set_time_sequence('homotopy', t)
        t += 1
        grasp = sim_grasps[i]
        cost_i = costs[i]
        geodesic_cost_i = get_geodesic_dist(get_grasp_locs(grasp.phy), goal.loc)
        print(f'cost: {cost_i:.3f} {grasp.locs=:} {grasp.strategy=:}, geodesic: {geodesic_cost_i:.3f}')
        rr.log_text_entry("planned", f'cost: {cost_i:.3f} grasp: {grasp.locs=:}')
        viz.viz(grasp.phy, is_planning=True)
    print(f'{initial_geodesic_cost=:.3f}')


if __name__ == "__main__":
    main()
