#!/usr/bin/env python3
import argparse
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import pybio_ik
import pysdf_tools
import rerun as rr
from colorama import Fore

import rospy
from arc_utilities import ros_init
from mjregrasping.goals import ThreadingGoal
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_grasp_locs
from mjregrasping.homotopy_regrasp_planner import release_and_settle, grasp_and_settle
from mjregrasping.homotopy_utils import load_skeletons
from mjregrasping.move_to_joint_config import execute_grasp_plan
from mjregrasping.movie import MjMovieMaker
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.regrasp_goal import GraspGoal, RegraspGoal
from mjregrasping.regrasp_mpc import RegraspMPC
from mjregrasping.regrasping_mppi import do_grasp_dynamics
from mjregrasping.rollout import control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import cable_harness, setup_cable_harness, make_ch_goal1
from mjregrasping.segment_demo import load_demo, get_subgoals_by_h
from mjregrasping.viz import make_viz


@ros_init.with_ros("cable_harness")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    parser = argparse.ArgumentParser()
    parser.add_argument('demo', type=Path)
    args = parser.parse_args()

    rr.init('regrasp_mpc_runner')
    rr.connect()

    scenario = cable_harness

    viz = make_viz(scenario)

    root = Path("results") / scenario.name
    root.mkdir(exist_ok=True, parents=True)

    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
    d = mujoco.MjData(m)
    objects = MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
    phy = Physics(m, d, objects)

    sdf = pysdf_tools.SignedDistanceField.LoadFromFile(str(scenario.sdf_path))
    # viz.sdf(sdf, '', 0)
    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)

    setup_cable_harness(phy, viz)

    mov = MjMovieMaker(m)
    now = int(time.time())
    seed = 0
    mov_path = root / f'seed_{seed}_{now}.mp4'
    print(f"Saving movie to {mov_path}")
    mov.start(mov_path, fps=8)

    # First, we need to segment the demonstration and construct a sequences of Goal instances based on that
    print("loading demo...")
    hs, locs, phys = load_demo(args.demo, scenario)

    skeletons = load_skeletons(scenario.skeletons_path)
    grasp_rrt = GraspRRT()
    subgoals = list(get_subgoals_by_h(phys, hs))
    subgoal_hs = [subgoal[1] for subgoal in subgoals]

    goals = []
    # for subgoal, next_subgoal in zip(subgoals, subgoals[1:]):
    #     locs, h = subgoal
    #     next_locs, next_h = next_subgoal
    #     is_next_grasp = (locs == -1) & (next_locs != -1)
    #     loc = locs[np.where(locs != -1)[0][0]]  # assumes there are two grippers that are alternating
    #     next_loc = next_locs[np.where(is_next_grasp)[0][0]]  # assumes there are two grippers that are alternating
    #     next_tool_name = phy.o.rd.tool_sites[np.where(is_next_grasp)[0][0]]
    #     skel = skeletons['loop3']
    #     goal_i = ThreadingGoal(skel, loc, next_tool_name, next_loc, bio_ik, viz)
    #     goals.append(goal_i)
    # goals.append(subgoals[-1])
    goals = [
        ThreadingGoal(skeletons, 'loop1', 0.93, 'left', 1.0, subgoal_hs[1], grasp_rrt, viz),
        ThreadingGoal(skeletons, 'loop1', 0.96, 'right', 0.96, subgoal_hs[3], grasp_rrt, viz),
    ]

    pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
    mpc = RegraspMPC(pool, phy.m.nu, load_skeletons(scenario.skeletons_path), sdf, seed, viz, mov)
    num_samples = hp['regrasp_n_samples']
    goal_idx = 0

    grasp_goal = GraspGoal(get_grasp_locs(phy))

    mpc.mppi.reset()
    mpc.reset_trap_detection()

    threading_goal = goals[goal_idx]
    regrasp_goal = RegraspGoal(threading_goal, grasp_goal, hp['grasp_goal_radius'], mpc.viz)

    itr = 0
    max_iters = 500
    mpc.viz.viz(phy)
    while True:
        if rospy.is_shutdown():
            mpc.mov.close()
            raise RuntimeError("ROS shutdown")

        if itr > max_iters:
            break

        regrasp_goal.viz_goal(phy)

        if regrasp_goal.satisfied(phy):
            print(Fore.GREEN + "SubGoal reached!" + Fore.RESET)
            mpc.mppi.reset()
            # grasp end with left gripper
            strategy = [Strategies.NEW_GRASP, Strategies.STAY]
            locs = np.array([1.0, 0.93])
            res = grasp_rrt.plan(phy, strategy, locs,
                                 None, True, allowed_planning_time=1.0)
            release_and_settle(phy, strategy, viz, is_planning=False, mov=mov)
            qs = np.array([p.positions for p in res.trajectory.joint_trajectory.points])
            execute_grasp_plan(phy, qs, viz, is_planning=False, mov=mov)
            grasp_and_settle(phy, locs, viz, is_planning=False, mov=mov)
            grasp_goal.set_grasp_locs(locs)

            # release right
            strategy = [Strategies.NEW_GRASP, Strategies.RELEASE]
            release_and_settle(phy, strategy, viz, is_planning=False, mov=mov)

            goal_idx += 1
            threading_goal = goals[goal_idx]
            regrasp_goal.op_goal = threading_goal
            if goal_idx >= len(goals):
                print(Fore.GREEN + "Goal reached!" + Fore.RESET)
                break

        command, sub_time_s = mpc.mppi.command(phy, regrasp_goal, num_samples, viz=mpc.viz)
        mpc.mppi_viz(mpc.mppi, regrasp_goal, phy, command, sub_time_s)

        control_step(phy, command, sub_time_s)
        mpc.viz.viz(phy)

        if mpc.mov:
            mpc.mov.render(phy.d)

        results = regrasp_goal.get_results(phy)
        do_grasp_dynamics(phy, results)

        mpc.mppi.roll()

        itr += 1

    mpc.close()


if __name__ == "__main__":
    main()
