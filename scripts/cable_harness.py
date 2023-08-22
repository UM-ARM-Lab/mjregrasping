#!/usr/bin/env python3
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
from mjregrasping.goals import ThreadingGoal, GraspLocsGoal, ObjectPointGoal
from mjregrasping.grasp_and_settle import release_and_settle, grasp_and_settle
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_grasp_locs
from mjregrasping.homotopy_utils import load_skeletons
from mjregrasping.movie import MjMovieMaker
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.regrasp_goal import RegraspGoal
from mjregrasping.regrasp_mpc import RegraspMPC
from mjregrasping.regrasping_mppi import do_grasp_dynamics
from mjregrasping.rollout import control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import cable_harness, setup_cable_harness
from mjregrasping.segment_demo import load_demo, get_subgoals_by_h
from mjregrasping.teleport_to_plan import teleport_to_end_of_plan
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
    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)

    setup_cable_harness(phy, viz)

    mov = MjMovieMaker(m)
    now = int(time.time())
    seed = 2
    mov_path = root / f'seed_{seed}_{now}.mp4'
    print(f"Saving movie to {mov_path}")
    mov.start(mov_path, fps=8)

    # First, we need to segment the demonstration and construct a sequences of Goal instances based on that
    print("loading demo...")
    hs, locs, phys, _ = load_demo(args.demo, scenario)

    skeletons = load_skeletons(scenario.skeletons_path)
    grasp_rrt = GraspRRT()
    subgoals = list(get_subgoals_by_h(phys, hs))
    subgoal_locs = [subgoal[0] for subgoal in subgoals]
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
        ThreadingGoal(skeletons, 'loop1', subgoal_locs[1][0], 'left', subgoal_locs[1], subgoal_hs[1], grasp_rrt, viz),
        ([Strategies.NEW_GRASP, Strategies.RELEASE], subgoal_locs[2]),
        ThreadingGoal(skeletons, 'loop1', subgoal_locs[3][1], 'right', subgoal_locs[3], subgoal_hs[3], grasp_rrt, viz),
        ([Strategies.RELEASE, Strategies.NEW_GRASP], subgoal_locs[4]),
        ThreadingGoal(skeletons, 'loop2', subgoal_locs[5][0], 'left', subgoal_locs[5], subgoal_hs[5], grasp_rrt, viz),
        ([Strategies.NEW_GRASP, Strategies.RELEASE], subgoal_locs[6]),
        ThreadingGoal(skeletons, 'loop2', subgoal_locs[7][1], 'right', subgoal_locs[7], subgoal_hs[7], grasp_rrt, viz),
        ([Strategies.RELEASE, Strategies.NEW_GRASP], subgoal_locs[8]),
        ThreadingGoal(skeletons, 'loop3', subgoal_locs[9][0], 'left', subgoal_locs[9], subgoal_hs[9], grasp_rrt, viz),
        ([Strategies.NEW_GRASP, Strategies.RELEASE], subgoal_locs[10]),
        ObjectPointGoal(np.array([-0.75, 0.75, 0.4]), 0.05, 1, viz)
    ]

    pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
    mpc = RegraspMPC(pool, phy.m.nu, load_skeletons(scenario.skeletons_path), sdf, seed, viz, mov)
    num_samples = hp['regrasp_n_samples']
    goal_idx = 0

    grasp_goal = GraspLocsGoal(get_grasp_locs(phy))

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

        res, scene_msg = threading_goal.plan_to_next_locs(phy)
        if res:
            if threading_goal.satisfied_from_res(phy, res):
                print(Fore.GREEN + "SubGoal reached!" + Fore.RESET)
                mpc.mppi.reset()
                grasp_rrt.display_result(viz, res, scene_msg)
                goal_idx += 1
                strategy, locs = goals[goal_idx]
                # qs = np.array([p.positions for p in res.trajectory.joint_trajectory.points])
                # execute_grasp_plan(phy, qs, viz, is_planning=False, mov=mov, stop_on_contact=True)
                teleport_to_end_of_plan(phy, res)
                # force the grippers to be closed, just like we do in `make_planning_scene()`
                teleport_grippers_closed(phy)
                viz.viz(phy, is_planning=False)
                grasp_and_settle(phy, locs, viz, is_planning=False, mov=mov)
                release_and_settle(phy, strategy, viz, is_planning=False, mov=mov)
                # TODO: confirm that the H signature of the grasp is correct.
                #  if it's not, then we need to replan the grasp.
                grasp_goal.set_grasp_locs(locs)

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


def teleport_grippers_closed(phy):
    gripper_q_ids = [phy.m.joint(n).qposadr[0] for n in phy.o.rd.gripper_joint_names]
    gripper_act_ids = [phy.m.actuator(n).trnid[0] for n in phy.o.rd.gripper_actuator_names]
    phy.d.qpos[gripper_q_ids] = hp['finger_q_closed']
    phy.d.qpos[gripper_act_ids] = hp['finger_q_closed']
    mujoco.mj_forward(phy.m, phy.d)


if __name__ == "__main__":
    main()
