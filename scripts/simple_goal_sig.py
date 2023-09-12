#!/usr/bin/env python3
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import mujoco
import numpy as np
import rerun as rr
from colorama import Fore

import rospy
from arc_utilities import ros_init
from mjregrasping.goals import GraspLocsGoal, point_goal_from_geom
from mjregrasping.grasp_and_settle import deactivate_moving, grasp_and_settle, deactivate_release
from mjregrasping.grasping import get_grasp_locs
from mjregrasping.homotopy_checker import through_skels
from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner
from mjregrasping.move_to_joint_config import pid_to_joint_configs
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.regrasping_mppi import RegraspMPPI, mppi_viz, do_grasp_dynamics
from mjregrasping.rollout import control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import simple_goal_sig
from mjregrasping.tamp_regrasp_planner import TAMPRegraspPlanner
from mjregrasping.trap_detection import TrapDetection
from mjregrasping.trials import load_trial
from mjregrasping.viz import make_viz, Viz
from moveit_msgs.msg import MoveItErrorCodes


@ros_init.with_ros("simple_goal_sig")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('simple_goal_sig')
    rr.connect()

    hp["tamp_horizon"] = 10

    scenario = simple_goal_sig

    viz = make_viz(scenario)

    gl_ctx = mujoco.GLContext(1280, 720)
    gl_ctx.make_current()

    grasp_rrt = GraspRRT()
    planning_times = []

    for i in range(0, 25):
        phy, sdf, skeletons, mov = load_trial(i, gl_ctx, scenario, viz)

        grasp_goal = GraspLocsGoal(get_grasp_locs(phy))

        pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
        mppi = RegraspMPPI(pool=pool, nu=phy.m.nu, seed=i, horizon=hp['horizon'], noise_sigma=scenario.noise_sigma,
                           temp=hp['temp'])
        num_samples = hp['n_samples']

        mppi.reset()

        mpc_times = []
        overall_t0 = perf_counter()

        end_loc = 0.98
        goal = point_goal_from_geom(grasp_goal, phy, "goal", end_loc, viz)
        goal_skel_names = []  # Empty list means m={[0]}, aka not through any skeletons

        traps = TrapDetection()

        itr = 0

        viz.viz(phy)
        success = False
        while True:
            if rospy.is_shutdown():
                mov.close()
                break

            if itr > 75:
                print(Fore.RED + "Max iterations reached!" + Fore.RESET)
                break

            goal.viz_goal(phy)

            if goal.satisfied(phy) and through_skels(skeletons, goal_skel_names, phy):
                print(Fore.GREEN + f"Task complete! {success=}" + Fore.RESET)
                break
            if traps.check_is_stuck(phy, grasp_goal):
                # planner = HomotopyRegraspPlanner(goal.loc, grasp_rrt, skeletons, goal_skel_names)
                planner = TAMPRegraspPlanner(scenario, goal, grasp_rrt, skeletons, None)

                planning_t0 = perf_counter()
                sim_grasps = planner.simulate_sampled_grasps(phy, viz, viz_execution=True)
                best_grasp = planner.get_best(sim_grasps, viz=viz)
                planning_times.append(perf_counter() - planning_t0)
                viz.viz(best_grasp.phy, is_planning=True)
                if best_grasp.res.error_code.val == MoveItErrorCodes.SUCCESS:
                    execute_grasp_change_plan(best_grasp, goal.grasp_goal, phy, viz, mov)
                    traps.reset_trap_detection()
                else:
                    print("No plans found!")

            mpc_t0 = perf_counter()
            command, sub_time_s = mppi.command(phy, goal, num_samples, viz=viz)
            mpc_times.append(perf_counter() - mpc_t0)
            mppi_viz(mppi, goal, phy, command, sub_time_s)

            control_step(phy, command, sub_time_s, mov=mov)
            viz.viz(phy)

            do_grasp_dynamics(phy)

            mppi.roll()

            itr += 1

        # save the results
        metrics = {
            'itr':            itr,
            'success':        success,
            'sim_time':       phy.d.time,
            'planning_times': planning_times,
            'mpc_times':      mpc_times,
            'overall_time':   perf_counter() - overall_t0,
            'grasp_history':  np.array(grasp_goal.history).tolist(),
            'method':         '\\signature{}',
            'hp':             hp,
        }
        mov.close(metrics)


def execute_grasp_change_plan(best_grasp, grasp_goal: GraspLocsGoal, phy: Physics, viz: Viz, mov):
    deactivate_moving(phy, best_grasp.strategy, viz, is_planning=False, mov=mov)
    pid_to_joint_configs(phy, best_grasp.res, viz, is_planning=False, mov=mov)
    grasp_and_settle(phy, best_grasp.locs, viz, is_planning=False, mov=mov)
    deactivate_release(phy, best_grasp.strategy, viz=viz, is_planning=False, mov=mov)
    grasp_goal.set_grasp_locs(best_grasp.locs)
    print(f"Changed grasp to {best_grasp.locs}")


if __name__ == "__main__":
    main()
