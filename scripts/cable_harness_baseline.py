#!/usr/bin/env python3
""""
Based on "An Online Method for Tight-tolerance Insertion Tasks for String and Rope" by Wang, Berenson, and Balkcom, 2015
"""
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rerun as rr
from colorama import Fore

import rospy
from arc_utilities import ros_init
from mjregrasping.goals import GraspLocsGoal, ObjectPointGoal, point_goal_from_geom, \
    WeifuThreadingGoal
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_grasp_locs, activate_grasp
from mjregrasping.homotopy_utils import skeleton_field_dir
from mjregrasping.params import hp
from mjregrasping.trials import load_trial, save_metrics
from mjregrasping.regrasping_mppi import do_grasp_dynamics, RegraspMPPI, mppi_viz
from mjregrasping.rollout import control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import cable_harness
from mjregrasping.teleport_to_plan import teleport_to_planned_q
from mjregrasping.viz import make_viz


@ros_init.with_ros("cable_harness_baseline")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('cable_harness_baseline')
    rr.connect()

    scenario = cable_harness

    viz = make_viz(scenario)

    grasp_rrt = GraspRRT()

    for i in range(10):
        phy, sdf, skeletons, mov, metrics_path = load_trial(i, scenario, viz)

        grasp_goal = GraspLocsGoal(get_grasp_locs(phy))

        pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
        mppi = RegraspMPPI(pool=pool, nu=phy.m.nu, seed=i, horizon=hp['horizon'],
                           noise_sigma=cable_harness.noise_sigma,
                           temp=hp['temp'])
        num_samples = hp['n_samples']
        goal_idx = 0

        mppi.reset()

        end_loc = 0.98
        goals = [
            WeifuThreadingGoal(grasp_goal, skeletons, 'loop1', end_loc, sdf, viz),
            WeifuThreadingGoal(grasp_goal, skeletons, 'loop2', end_loc, sdf, viz),
            WeifuThreadingGoal(grasp_goal, skeletons, 'loop3', end_loc, sdf, viz),
            point_goal_from_geom(grasp_goal, phy, "goal", 1, viz)
        ]
        goal = goals[goal_idx]

        itr = 0

        viz.viz(phy)
        success = False
        while True:
            if rospy.is_shutdown():
                mov.close()
                break

            if itr > scenario.max_iters:
                print(Fore.RED + "Max iterations reached!" + Fore.RESET)
                break

            goal.viz_goal(phy)

            if isinstance(goal, ObjectPointGoal):
                if goal.satisfied(phy):
                    print(Fore.GREEN + "Goal reached!" + Fore.RESET)
                    break
            else:
                disc_center = np.mean(goal.skel[:4], axis=0)
                disc_normal = skeleton_field_dir(goal.skel, disc_center[None])[0] * 0.01
                disc_rad = 0.05  # TODO: compute the radius of the disc

                if goal.satisfied(phy, disc_center, disc_normal, disc_rad):
                    mppi.reset()

                    strategy = [Strategies.STAY, Strategies.MOVE]
                    locs = np.array([-1, end_loc])
                    res, scene_msg = grasp_rrt.plan(phy, strategy, locs, viz)
                    if res.error_code.val == res.error_code.SUCCESS:
                        print("Moving on to next goal")
                        grasp_rrt.display_result(viz, res, scene_msg)
                        qs = np.array([p.positions for p in res.trajectory.joint_trajectory.points])
                        teleport_to_planned_q(phy, qs[-1])
                        goal.loc = end_loc
                        goal_idx += 1
                    else:
                        print("Lowering grasp")
                        goal.loc -= 0.01
                    activate_grasp(phy, 'right', goal.loc)
                    control_step(phy, np.zeros(phy.m.nu), sub_time_s=1.0, mov=mov)
                    viz.viz(phy, False)
                    goal = goals[goal_idx]
                    grasp_goal.set_grasp_locs(np.array([-1, goal.loc]))

                    if goal_idx >= len(goals):
                        print(Fore.GREEN + "Task complete!" + Fore.RESET)
                        success = True
                        break

            command, sub_time_s = mppi.command(phy, goal, num_samples, viz=viz)
            mppi_viz(viz, mppi, goal, phy, command, sub_time_s)

            control_step(phy, command, sub_time_s, mov=mov)
            viz.viz(phy)

            results = goal.get_results(phy)
            do_grasp_dynamics(phy, results)

            mppi.roll()

            itr += 1

        save_metrics(metrics_path, mov, itr=itr, success=success, time=phy.d.time)


if __name__ == "__main__":
    main()
