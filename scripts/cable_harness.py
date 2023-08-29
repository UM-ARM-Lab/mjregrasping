#!/usr/bin/env python3
import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rerun as rr
from colorama import Fore
from multiset import Multiset

import rospy
from arc_utilities import ros_init
from mjregrasping.goals import ThreadingGoal, GraspLocsGoal, ObjectPointGoal, PullThroughGoal, point_goal_from_geom
from mjregrasping.grasp_and_settle import release_and_settle, grasp_and_settle
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_grasp_locs
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.params import hp
from mjregrasping.trials import load_trial, save_metrics
from mjregrasping.regrasping_mppi import do_grasp_dynamics, RegraspMPPI, mppi_viz
from mjregrasping.rollout import control_step, DEFAULT_SUB_TIME_S
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import cable_harness
from mjregrasping.teleport_to_plan import teleport_along_plan
from mjregrasping.trap_detection import TrapDetection
from mjregrasping.viz import make_viz


@ros_init.with_ros("cable_harness")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('cable_harness')
    rr.connect()

    scenario = cable_harness

    grasp_rrt = GraspRRT()
    subgoal_locs = [
        np.array([-1, 0.93]),
        np.array([1.0, 0.93]),
        np.array([1.0, -1]),
        np.array([1.0, 0.95]),  # 3
        np.array([-1.0, 0.95]),
        np.array([1.0, 0.95]),
        np.array([1.0, -1]),
        np.array([1.0, 0.95]),
        np.array([-1.0, 0.95]),
        np.array([1.0, 0.95]),
        np.array([1.0, -1]),
    ]
    subgoal_hs = [
        Multiset([('a0,b,g1', 0, 0, 0)]),
        Multiset([('b,g0,g1', 1, 0, 0), ('a0,b,g1', 0, 0, 0)]),
        Multiset([('a0,b,g0', 1, 0, 0)]),
        Multiset([('b,g0,g1', 0, 0, 0), ('a0,b,g1', 1, 0, 0)]),  # 3
        Multiset([('a0,b,g1', 1, 0, 0)]),
        Multiset([('b,g0,g1', 0, 1, 0), ('a0,b,g1', 1, 0, 0)]),
        Multiset([('a0,b,g0', 1, 1, 0)]),
        Multiset([('b,g0,g1', 0, 0, 0), ('a0,b,g1', 1, 1, 0)]),
        Multiset([('a0,b,g1', 1, 1, 0)]),
        Multiset([('b,g0,g1', 0, 0, 1), ('a0,b,g1', 1, 1, 0)]),
    ]

    viz = make_viz(scenario)
    for i in range(1, 10):
        phy, sdf, skeletons, mov, metrics_path = load_trial(i, scenario, viz)

        grasp_goal = GraspLocsGoal(get_grasp_locs(phy))

        goals = [
            ThreadingGoal(grasp_goal, skeletons, 'loop1', subgoal_locs[1][0], 'left', subgoal_locs[1], subgoal_hs[1],
                          grasp_rrt, sdf, viz),
            ([Strategies.NEW_GRASP, Strategies.RELEASE], subgoal_locs[2]),
            PullThroughGoal(grasp_goal, skeletons, 'loop1', subgoal_locs[3][1], 'right', subgoal_locs[3], subgoal_hs[3],
                            grasp_rrt, sdf, viz),
            ([Strategies.RELEASE, Strategies.NEW_GRASP], subgoal_locs[4]),
            ThreadingGoal(grasp_goal, skeletons, 'loop2', subgoal_locs[5][0], 'left', subgoal_locs[5], subgoal_hs[5],
                          grasp_rrt, sdf, viz),
            ([Strategies.NEW_GRASP, Strategies.RELEASE], subgoal_locs[6]),
            PullThroughGoal(grasp_goal, skeletons, 'loop2', subgoal_locs[7][1], 'right', subgoal_locs[7], subgoal_hs[7],
                            grasp_rrt, sdf, viz),
            ([Strategies.RELEASE, Strategies.NEW_GRASP], subgoal_locs[8]),
            ThreadingGoal(grasp_goal, skeletons, 'loop3', subgoal_locs[9][0], 'left', subgoal_locs[9], subgoal_hs[9],
                          grasp_rrt, sdf, viz),
            ([Strategies.NEW_GRASP, Strategies.RELEASE], subgoal_locs[10]),
            point_goal_from_geom(grasp_goal, phy, "goal", 1, viz)
        ]

        pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
        traps = TrapDetection()
        mppi = RegraspMPPI(pool=pool, nu=phy.m.nu, seed=i, horizon=hp['horizon'],
                           noise_sigma=cable_harness.noise_sigma,
                           temp=hp['temp'])
        num_samples = hp['n_samples']
        goal_idx = 0

        mppi.reset()
        traps.reset_trap_detection()

        goal = goals[goal_idx]

        itr = 0
        success = False
        viz.viz(phy)
        while True:
            if rospy.is_shutdown():
                mov.close()
                break

            if itr >= scenario.max_iters:
                break

            goal.viz_goal(phy)

            if isinstance(goal, ObjectPointGoal):
                if goal.satisfied(phy):
                    print(Fore.GREEN + "Task complete!" + Fore.RESET)
                    success = True
                    break
            else:
                res, scene_msg = goal.plan_to_next_locs(phy)
                if res:
                    if goal.satisfied_from_res(phy, res):
                        print(Fore.GREEN + "SubGoal reached!" + Fore.RESET)
                        mppi.reset()
                        grasp_rrt.display_result(viz, res, scene_msg)
                        goal_idx += 1
                        strategy, locs = goals[goal_idx]
                        qs = np.array([p.positions for p in res.trajectory.joint_trajectory.points])
                        # force the grippers to be closed, just like we do in `make_planning_scene()`
                        pid_to_joint_config(phy, viz, qs[0], DEFAULT_SUB_TIME_S, is_planning=False, mov=mov)
                        teleport_along_plan(phy, qs, viz, is_planning=False, mov=mov)
                        # execute_grasp_plan(phy, qs, viz, is_planning=False, mov=mov, stop_on_contact=True)
                        viz.viz(phy, is_planning=False)
                        grasp_and_settle(phy, locs, viz, is_planning=False, mov=mov)
                        release_and_settle(phy, strategy, viz, is_planning=False, mov=mov)
                        # TODO: confirm that the H signature of the grasp is correct.
                        #  if it's not, then we need to replan the grasp.
                        grasp_goal.set_grasp_locs(locs)

                        goal_idx += 1
                        goal = goals[goal_idx]

            command, sub_time_s = mppi.command(phy, goal, num_samples, viz=viz)
            mppi_viz(viz, mppi, goal, phy, command, sub_time_s)

            control_step(phy, command, sub_time_s, mov=mov)
            viz.viz(phy)

            results = goal.get_results(phy)
            do_grasp_dynamics(phy, results)

            mppi.roll()

            itr += 1

        # save the results
        save_metrics(metrics_path, mov, itr=itr, success=success, time=phy.d.time)


if __name__ == "__main__":
    main()
