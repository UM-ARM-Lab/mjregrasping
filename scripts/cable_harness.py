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
from mjregrasping.goals import ThreadingGoal, GraspLocsGoal, ObjectPointGoal, PullThroughGoal, point_goal_from_geom
from mjregrasping.grasp_and_settle import release_and_settle, grasp_and_settle
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_grasp_locs
from mjregrasping.homotopy_utils import load_skeletons
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.movie import MjMovieMaker
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.trap_detection import TrapDetection
from mjregrasping.regrasping_mppi import do_grasp_dynamics, RegraspMPPI, mppi_viz
from mjregrasping.rerun_visualizer import log_skeletons
from mjregrasping.robot_data import val
from mjregrasping.rollout import control_step, DEFAULT_SUB_TIME_S
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import cable_harness, setup_cable_harness
from mjregrasping.segment_demo import get_subgoals_by_h, load_cached_demo
from mjregrasping.teleport_to_plan import teleport_to_end_of_plan, teleport_along_plan
from mjregrasping.viz import make_viz


@ros_init.with_ros("cable_harness")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    parser = argparse.ArgumentParser()
    parser.add_argument('demo', type=Path)
    args = parser.parse_args()

    rr.init('cable_harness')
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
    seed = 1
    mov_path = root / f'seed_{seed}_{now}.mp4'
    print(f"Saving movie to {mov_path}")
    mov.start(mov_path)

    # First, we need to segment the demonstration and construct a sequences of Goal instances based on that
    print("loading demo...")
    hs, locs, phys, _ = load_cached_demo(args.demo)
    print("done.")

    skeletons = load_skeletons(scenario.skeletons_path)
    log_skeletons(skeletons)
    grasp_rrt = GraspRRT()
    subgoals = list(get_subgoals_by_h(phys, hs))
    subgoal_locs = [subgoal[0] for subgoal in subgoals]
    subgoal_hs = [subgoal[1] for subgoal in subgoals]

    grasp_goal = GraspLocsGoal(get_grasp_locs(phy))

    goals = [
        ThreadingGoal(grasp_goal, skeletons, 'loop1', subgoal_locs[1][0], 'left', subgoal_locs[1], subgoal_hs[1],
                      grasp_rrt, sdf,
                      viz),
        ([Strategies.NEW_GRASP, Strategies.RELEASE], subgoal_locs[2]),
        PullThroughGoal(grasp_goal, skeletons, 'loop1', subgoal_locs[3][1], 'right', subgoal_locs[3], subgoal_hs[3],
                        grasp_rrt, sdf,
                        viz),
        ([Strategies.RELEASE, Strategies.NEW_GRASP], subgoal_locs[4]),
        ThreadingGoal(grasp_goal, skeletons, 'loop2', subgoal_locs[5][0], 'left', subgoal_locs[5], subgoal_hs[5],
                      grasp_rrt, sdf,
                      viz),
        ([Strategies.NEW_GRASP, Strategies.RELEASE], subgoal_locs[6]),
        PullThroughGoal(grasp_goal, skeletons, 'loop2', subgoal_locs[7][1], 'right', subgoal_locs[7], subgoal_hs[7],
                        grasp_rrt, sdf,
                        viz),
        ([Strategies.RELEASE, Strategies.NEW_GRASP], subgoal_locs[8]),
        ThreadingGoal(grasp_goal, skeletons, 'loop3', subgoal_locs[9][0], 'left', subgoal_locs[9], subgoal_hs[9],
                      grasp_rrt, sdf,
                      viz),
        ([Strategies.NEW_GRASP, Strategies.RELEASE], subgoal_locs[10]),
        point_goal_from_geom(grasp_goal, phy, "goal", 1, viz)
    ]

    pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
    traps = TrapDetection()
    mppi = RegraspMPPI(pool=pool, nu=phy.m.nu, seed=seed, horizon=hp['horizon'], noise_sigma=val.noise_sigma,
                       temp=hp['temp'])
    num_samples = hp['n_samples']
    goal_idx = 0

    mppi.reset()
    traps.reset_trap_detection()

    goal = goals[goal_idx]

    itr = 0
    max_iters = 500
    viz.viz(phy)
    while True:
        if rospy.is_shutdown():
            mov.close()
            break

        if itr > max_iters:
            break

        goal.viz_goal(phy)

        if isinstance(goal, ObjectPointGoal):
            if goal.satisfied(phy):
                print(Fore.GREEN + "Goal reached!" + Fore.RESET)
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
                    # teleport_grippers_pregrasp(phy)
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
                    if goal_idx >= len(goals):
                        print(Fore.GREEN + "Goal reached!" + Fore.RESET)
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


def teleport_grippers_pregrasp(phy):
    gripper_q_ids = [phy.m.joint(n).qposadr[0] for n in phy.o.rd.gripper_joint_names]
    gripper_act_ids = [phy.m.actuator(n).trnid[0] for n in phy.o.rd.gripper_actuator_names]
    phy.d.qpos[gripper_q_ids] = hp['finger_q_pregrasp']
    phy.d.qpos[gripper_act_ids] = hp['finger_q_pregrasp']
    mujoco.mj_forward(phy.m, phy.d)


if __name__ == "__main__":
    main()
