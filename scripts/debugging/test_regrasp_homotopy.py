#!/usr/bin/env python3
from time import perf_counter

import mujoco
import numpy as np
import rerun as rr

import rospy
from arc_utilities import ros_init
from mjregrasping.goals import point_goal_from_geom, GraspLocsGoal
from mjregrasping.grasping import get_grasp_locs
from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner, get_geodesic_dist
from mjregrasping.move_to_joint_config import execute_grasp_plan
from mjregrasping.physics import get_q
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
    planner = HomotopyRegraspPlanner(goal, grasp_rrt, skeletons, seed=i)

    initial_geodesic_cost = get_geodesic_dist(get_grasp_locs(phy), goal)

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

    errors = 0
    for i in range(25):
        phy_i = phy.copy_all()

        from mjregrasping.grasp_strategies import Strategies
        strategy = [Strategies.NEW_GRASP, Strategies.STAY]
        candidate_locs = np.array([0.99, 0.93])

        res, scene_msg = grasp_rrt.plan(phy_i, strategy, candidate_locs, viz, pos_noise=0.05)
        print(len(res.trajectory.joint_trajectory.points))
        grasp_rrt.display_result(viz, res, scene_msg)

        qs = np.array([p.positions for p in res.trajectory.joint_trajectory.points])
        if len(qs) == 0:
            continue

        execute_grasp_plan(phy_i, qs, viz, is_planning=False, mov=None)

        final_q = get_q(phy_i)
        error = np.abs(final_q - qs[-1]).max()
        if error > np.deg2rad(3):
            errors += 1
            print("Error!")

    return
    grasps_inputs = planner.sample_grasp_inputs(phy)
    t0 = perf_counter()
    sim_grasps = planner.simulate_grasps(grasps_inputs, phy, viz=viz, viz_execution=True)
    t1 = perf_counter()
    print(f'planning time: {t1 - t0:.3f}s')
    best_grasp = planner.get_best(sim_grasps, viz=None)

    costs = [planner.cost(g) for g in sim_grasps]
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
    costs = [planner.cost(g) for g in sim_grasps]
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
