#!/usr/bin/env python3
""""
Based on "An Online Method for Tight-tolerance Insertion Tasks for String and Rope" by Wang, Berenson, and Balkcom, 2015
"""
import argparse
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import pysdf_tools
import rerun as rr
from colorama import Fore

import rospy
from arc_utilities import ros_init
from mjregrasping.goals import GraspLocsGoal, ObjectPointGoal, point_goal_from_geom, \
    WeifuThreadingGoal
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_grasp_locs, activate_grasp
from mjregrasping.homotopy_utils import skeleton_field_dir
from mjregrasping.movie import MjMovieMaker
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.regrasping_mppi import do_grasp_dynamics, RegraspMPPI, mppi_viz
from mjregrasping.rerun_visualizer import log_skeletons
from mjregrasping.rollout import control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import cable_harness, setup_cable_harness, get_cable_harness_skeletons
from mjregrasping.teleport_to_plan import teleport_to_planned_q
from mjregrasping.viz import make_viz


@ros_init.with_ros("cable_harness_baseline")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    rr.init('cable_harness_baseline')
    rr.connect()

    scenario = cable_harness

    viz = make_viz(scenario)

    root = Path("results") / scenario.name + "_baseline"
    root.mkdir(exist_ok=True, parents=True)

    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
    d = mujoco.MjData(m)
    objects = MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
    phy = Physics(m, d, objects)

    sdf = pysdf_tools.SignedDistanceField.LoadFromFile(str(scenario.sdf_path))
    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)

    setup_cable_harness(phy, viz)

    mov = MjMovieMaker(m)
    now = int(time.time())
    seed = 1
    mov_path = root / f'seed_{seed}_{now}.mp4'
    print(f"Saving movie to {mov_path}")
    mov.start(mov_path)

    log_skeletons(skeletons)

    grasp_rrt = GraspRRT()

    grasp_goal = GraspLocsGoal(get_grasp_locs(phy))

    pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
    mppi = RegraspMPPI(pool=pool, nu=phy.m.nu, seed=seed, horizon=hp['horizon'], noise_sigma=cable_harness.noise_sigma,
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

    max_iters = 250
    viz.viz(phy)
    while True:
        if rospy.is_shutdown():
            mov.close()
            break

        if itr > max_iters:
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
                    break

        command, sub_time_s = mppi.command(phy, goal, num_samples, viz=viz)
        mppi_viz(viz, mppi, goal, phy, command, sub_time_s)

        control_step(phy, command, sub_time_s, mov=mov)
        viz.viz(phy)

        results = goal.get_results(phy)
        do_grasp_dynamics(phy, results)

        mppi.roll()

        itr += 1

    mov.close()


if __name__ == "__main__":
    main()
