#!/usr/bin/env python3
import multiprocessing
import time
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import pysdf_tools
import rerun as rr
from colorama import Fore

import rospy
from arc_utilities import ros_init
from mjregrasping.goals import GraspLocsGoal, point_goal_from_geom
from mjregrasping.grasp_and_settle import release_and_settle, grasp_and_settle
from mjregrasping.grasping import get_grasp_locs
from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner, get_geodesic_dist
from mjregrasping.homotopy_utils import load_skeletons
from mjregrasping.move_to_joint_config import execute_grasp_plan
from mjregrasping.movie import MjMovieMaker
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.trap_detection import TrapDetection
from mjregrasping.regrasping_mppi import do_grasp_dynamics, RegraspMPPI, mppi_viz
from mjregrasping.robot_data import val
from mjregrasping.rollout import control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.sdf_collision_checker import SDFCollisionChecker
from mjregrasping.viz import make_viz
from moveit_msgs.msg import MoveItErrorCodes


@ros_init.with_ros("pulling")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('pulling')
    rr.connect()

    scenario = val_pulling
    viz = make_viz(scenario)

    root = Path("results") / scenario.name
    root.mkdir(exist_ok=True, parents=True)

    seed = 0
    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
    d = mujoco.MjData(m)
    objects = MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
    phy = Physics(m, d, objects)

    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)

    skeletons = {}
    # setup_pulling(phy, viz)

    grasp_goal = GraspLocsGoal(get_grasp_locs(phy))
    # Subtract a small amount from the radius so the rope is more clearly "inside" the goal region
    goal = point_goal_from_geom(grasp_goal, phy, "goal", 1, viz)

    mov = MjMovieMaker(m)
    now = int(time.time())
    mov_path = root / f'seed_{seed}_{now}.mp4'
    print(f"Saving movie to {mov_path}")
    mov.start(mov_path)

    pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
    traps = TrapDetection()
    mppi = RegraspMPPI(pool=pool, nu=phy.m.nu, seed=seed, horizon=hp['horizon'], noise_sigma=scenario.noise_sigma,
                       temp=hp['temp'])
    num_samples = hp['n_samples']
    grasp_rrt = GraspRRT()

    planner = HomotopyRegraspPlanner(goal, grasp_rrt, skeletons, None)

    goal.viz_goal(phy)

    mppi.reset()
    traps.reset_trap_detection()

    itr = 0
    max_iters = 100
    command = None
    sub_time_s = None
    viz.viz(phy)
    while True:
        if rospy.is_shutdown():
            mov.close()
            return False

        if itr > max_iters:
            break

        goal.viz_goal(phy)
        if goal.satisfied(phy):
            print(Fore.GREEN + "Goal reached!" + Fore.RESET)
            break

        is_stuck = traps.check_is_stuck(phy)
        needs_reset = False
        if is_stuck:
            print(Fore.YELLOW + "Stuck! Replanning..." + Fore.RESET)
            initial_geodesic_cost = get_geodesic_dist(grasp_goal.get_grasp_locs(), goal)
            sim_grasps = planner.simulate_sampled_grasps(phy, viz, viz_execution=False)
            best_grasp = planner.get_best(sim_grasps, viz=None)
            new_geodesic_cost = get_geodesic_dist(best_grasp.locs, goal)
            # if we are unable to improve by grasping closer to the keypoint, update the blacklist and replan
            if initial_geodesic_cost - new_geodesic_cost < 0.01:  # less than 1% closer to the keypoint
                print(Fore.YELLOW + "Unable to improve by grasping closer to the keypoint." + Fore.RESET)
                print(Fore.YELLOW + "Updating blacklist and replanning..." + Fore.RESET)
                planner.update_blacklists(phy)
                best_grasp = planner.get_best(sim_grasps, viz=None)
            if best_grasp.res.error_code.val != MoveItErrorCodes.SUCCESS:
                print(Fore.RED + "Failed to find a plan." + Fore.RESET)
            viz.viz(best_grasp.phy, is_planning=True)
            # now execute the plan
            release_and_settle(phy, best_grasp.strategy, viz, is_planning=False, mov=mov)
            qs = np.array([p.positions for p in best_grasp.res.trajectory.joint_trajectory.points])
            execute_grasp_plan(phy, qs, viz, is_planning=False, mov=mov)
            grasp_and_settle(phy, best_grasp.locs, viz, is_planning=False, mov=mov)
            grasp_goal.set_grasp_locs(best_grasp.locs)

            # save_data_and_eq(phy, Path(f'states/CableHarness/stuck1.pkl'))
            needs_reset = True

        n_warmstart = max(1, min(hp['warmstart'], int((1 - stuck_frac) * 5)))

        if needs_reset:
            mppi.reset()
            traps.reset_trap_detection()
            n_warmstart = hp['warmstart']

        for k in range(n_warmstart):
            command, sub_time_s = mppi.command(phy, goal, num_samples, viz=viz)
            mppi_viz(mppi, goal, phy, command, sub_time_s)

        control_step(phy, command, sub_time_s, mov=mov)
        viz.viz(phy)

        results = goal.get_results(phy)
        do_grasp_dynamics(phy, results)

        mppi.roll()

        itr += 1
    mov.close()


if __name__ == "__main__":
    main()
