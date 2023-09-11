#!/usr/bin/env python3
# NOTE: can't call this file threading.py because it conflicts with the threading module
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Dict

import mujoco
import numpy as np
import rerun as rr
from colorama import Fore

import rospy
from arc_utilities import ros_init
from mjregrasping.goals import GraspLocsGoal, ObjectPointGoal, point_goal_from_geom, ThreadingGoal, get_disc_params
from mjregrasping.grasp_and_settle import deactivate_moving, grasp_and_settle, deactivate_release
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_grasp_locs, get_is_grasping
from mjregrasping.homotopy_checker import through_skels
from mjregrasping.homotopy_regrasp_planner import HomotopyThreadingPlanner
from mjregrasping.move_to_joint_config import pid_to_joint_configs
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.regrasp_planner_utils import SimGraspCandidate
from mjregrasping.regrasping_mppi import RegraspMPPI, mppi_viz
from mjregrasping.rollout import control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import threading
from mjregrasping.teleport_to_plan import teleport_to_end_of_plan
from mjregrasping.trap_detection import TrapDetection
from mjregrasping.trials import load_trial
from mjregrasping.viz import make_viz
from moveit_msgs.msg import MoveItErrorCodes


@ros_init.with_ros("threading")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('threading')
    rr.connect()

    scenario = threading
    hp["threading_n_samples"] = 10

    viz = make_viz(scenario)

    gl_ctx = mujoco.GLContext(1280, 720)
    gl_ctx.make_current()

    grasp_rrt = GraspRRT()

    for i in range(0, 10):
        phy, sdf, skeletons, mov = load_trial(i, gl_ctx, scenario, viz)

        grasp_goal = GraspLocsGoal(get_grasp_locs(phy))

        pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
        mppi = RegraspMPPI(pool=pool, nu=phy.m.nu, seed=i, horizon=hp['horizon'],
                           noise_sigma=threading.noise_sigma,
                           temp=hp['temp'])
        num_samples = hp['n_samples']
        goal_idx = 0

        mppi.reset()

        mpc_times = []
        overall_t0 = perf_counter()

        end_loc = 1.0
        goals = [
            ThreadingGoal(grasp_goal, skeletons, ['loop1'], end_loc, viz),
            ThreadingGoal(grasp_goal, skeletons, ['loop1', 'loop2'], end_loc, viz),
            ThreadingGoal(grasp_goal, skeletons, ['loop1', 'loop2', 'loop3'], end_loc, viz),
            point_goal_from_geom(grasp_goal, phy, "goal", 1, viz)
        ]
        goal = goals[goal_idx]

        traps = TrapDetection()

        method = ThreadingMethodOurs(grasp_rrt, skeletons, traps, end_loc)
        print(f"Running method {method.__class__.__name__}")
        # method = ThreadingMethodWang(grasp_rrt, skeletons, traps, end_loc)

        itr = 0

        viz.viz(phy)
        success = False
        while True:
            if rospy.is_shutdown():
                mov.close()
                break

            if itr > 275:
                print(Fore.RED + "Max iterations reached!" + Fore.RESET)
                break

            goal.viz_goal(phy)

            if isinstance(goal, ObjectPointGoal):
                if goal.satisfied(phy):
                    success = False
                    print(Fore.GREEN + "Task complete!" + Fore.RESET)
                    break
            else:
                disc_center, disc_normal = get_disc_params(goal)
                disc_rad = 0.05  # TODO: compute the radius of the disc

                disc_penetrated = goal.satisfied(phy, disc_center, disc_normal, disc_rad)
                is_stuck = traps.check_is_stuck(phy)
                if disc_penetrated:
                    print("Disc penetrated!")
                    mppi.reset()
                    method.on_disc(phy, goal, grasp_goal, viz, mov)
                elif is_stuck:
                    print("Stuck!")
                    mppi.reset()
                    method.on_stuck(phy, goal, grasp_goal, viz, mov)

                if method.goal_satisfied(goal, phy):
                    goal_idx += 1
                    goal = goals[goal_idx]

            mpc_t0 = perf_counter()
            command, sub_time_s = mppi.command(phy, goal, num_samples, viz=viz)
            mpc_times.append(perf_counter() - mpc_t0)
            mppi_viz(mppi, goal, phy, command, sub_time_s)

            control_step(phy, command, sub_time_s, mov=mov)
            viz.viz(phy)

            # do_grasp_dynamics(phy)

            mppi.roll()

            itr += 1

        # save the results
        metrics = {
            'itr':            itr,
            'success':        success,
            'sim_time':       phy.d.time,
            'planning_times': method.planning_times,
            'mpc_times':      mpc_times,
            'overall_time':   perf_counter() - overall_t0,
            'grasp_history':  np.array(grasp_goal.history).tolist(),
            'method':         method.__class__.__name__,
        }
        mov.close(metrics)


class ThreadingMethod:

    def __init__(self, grasp_rrt: GraspRRT, skeletons: Dict, traps: TrapDetection, end_loc: float):
        self.grasp_rrt = grasp_rrt
        self.skeletons = skeletons
        self.traps = traps
        self.end_loc = end_loc
        self.planning_times = []

    def on_disc(self, phy, goal, grasp_goal, viz, mov):
        raise NotImplementedError()

    def on_stuck(self, phy, goal, grasp_goal, viz, mov):
        raise NotImplementedError()

    def goal_satisfied(self, goal, phy):
        raise NotImplementedError()


class ThreadingMethodWang(ThreadingMethod):
    """
    Based on "An Online Method for Tight-tolerance Insertion Tasks for String and Rope"
    by Wang, Berenson, and Balkcom. 2015
    """

    def __init__(self, grasp_rrt: GraspRRT, skeletons: Dict, traps: TrapDetection, end_loc: int):
        super().__init__(grasp_rrt, skeletons, traps, end_loc)
        self.plan_to_end_found = False

    def on_disc(self, phy, goal, grasp_goal, viz, mov):
        self.plan_to_end_found = False

        self.grasp_rrt.fix_start_state_in_place(phy)

        # Try to find a grasp to end loc with one arm.
        # The original paper uses floating grippers that can teleport, so we could adapt that in two ways:
        # 1) teleport the gripper to the final q of the plan
        method_1 = False
        if method_1:
            strategy = [Strategies.STAY, Strategies.MOVE]
            locs = np.array([-1, self.end_loc])
            planning_t0 = perf_counter()
            res, scene_msg = self.grasp_rrt.plan(phy, strategy, locs, viz,
                                                 pos_noise=1e-3)  # This method cant handle noise
            if res.error_code.val == MoveItErrorCodes.SUCCESS:
                self.planning_times.append(perf_counter() - planning_t0)
                teleport_to_end_of_plan(phy, res)
                grasp_and_settle(phy, locs, viz, is_planning=False, mov=mov)
                deactivate_release(phy, strategy, viz=viz, is_planning=False, mov=mov)
                grasp_goal.set_grasp_locs(locs)
                self.traps.reset_trap_detection()
                self.plan_to_end_found = True
                return

            # If we fail, try to scootch down the rope
            locs = grasp_goal.get_grasp_locs()
            locs[1] -= hp['wang_scootch_fraction']
            for _ in range(5):
                res, scene_msg = self.grasp_rrt.plan(phy, strategy, locs, viz,
                                                     pos_noise=1e-3)  # This method cant handle noise
                if res.error_code.val == MoveItErrorCodes.SUCCESS:
                    self.planning_times.append(perf_counter() - planning_t0)
                    teleport_to_end_of_plan(phy, res)
                    grasp_and_settle(phy, locs, viz, is_planning=False, mov=mov)
                    deactivate_release(phy, strategy, viz=viz, is_planning=False, mov=mov)
                    grasp_goal.set_grasp_locs(locs)
                    goal.loc -= hp['wang_scootch_fraction']
                    return
        else:
            # 2) alternate grippers and actually move the arms
            strategy = []
            locs = []
            for g_i in get_is_grasping(phy):
                if g_i:
                    strategy.append(Strategies.RELEASE)
                    locs.append(-1)
                else:
                    strategy.append(Strategies.NEW_GRASP)
                    locs.append(self.end_loc)
            locs = np.array(locs)
            planning_t0 = perf_counter()
            res, scene_msg = self.grasp_rrt.plan(phy, strategy, locs, viz)
            if res.error_code.val == MoveItErrorCodes.SUCCESS:
                self.planning_times.append(perf_counter() - planning_t0)
                grasp = SimGraspCandidate(None, None, strategy, res, locs, None)
                execute_grasp_change_plan(grasp, grasp_goal, phy, viz, mov)
                self.traps.reset_trap_detection()
                self.plan_to_end_found = True
                return

            # If we fail, try to scootch down the rope
            goal.loc = max(goal.loc - hp['wang_scootch_fraction'], max(get_grasp_locs(phy)))
            strategy = []
            locs = []
            loc_i = np.min(np.abs(grasp_goal.get_grasp_locs())) - hp['wang_scootch_fraction']
            for g_i in get_is_grasping(phy):
                if g_i:
                    strategy.append(Strategies.RELEASE)
                    locs.append(-1)
                else:
                    strategy.append(Strategies.NEW_GRASP)
                    locs.append(loc_i)
            res, scene_msg = self.grasp_rrt.plan(phy, strategy, locs, viz)
            if res.error_code.val == MoveItErrorCodes.SUCCESS:
                self.planning_times.append(perf_counter() - planning_t0)
                grasp = SimGraspCandidate(None, None, strategy, res, locs, None)
                execute_grasp_change_plan(grasp, grasp_goal, phy, viz, mov)
                self.traps.reset_trap_detection()
                return

        self.planning_times.append(perf_counter() - planning_t0)
        print("No plans found!")

    def on_stuck(self, phy, goal, grasp_goal, viz, mov):
        pass

    def goal_satisfied(self, goal, phy):
        ret = self.plan_to_end_found
        self.plan_to_end_found = False
        return ret


class ThreadingMethodOurs(ThreadingMethod):

    def on_disc(self, phy, goal, grasp_goal, viz, mov):
        planner = HomotopyThreadingPlanner(self.end_loc, self.grasp_rrt, self.skeletons, goal.skeleton_names)
        print(f"Planning with {planner.key_loc=}...")
        planning_t0 = perf_counter()
        sim_grasps = planner.simulate_sampled_grasps(phy, None, viz_execution=False)
        best_grasp = planner.get_best(sim_grasps, viz=viz)
        self.planning_times.append(perf_counter() - planning_t0)
        viz.viz(best_grasp.phy, is_planning=True)
        if best_grasp.res.error_code.val == MoveItErrorCodes.SUCCESS:
            if planner.through_skels(best_grasp.phy):
                print("Executing grasp change plan")
                execute_grasp_change_plan(best_grasp, grasp_goal, phy, viz, mov)
                self.traps.reset_trap_detection()
            else:
                print("Not through the goal skeleton!")
        else:
            # if we've reached the goal but can't grasp the end, scootch down the rope
            # but don't scootch past where we are currently grasping it.
            goal.loc = max(goal.loc - hp['ours_scootch_fraction'], max(get_grasp_locs(phy)))
            print("No plans found!")

    def on_stuck(self, phy, goal, grasp_goal, viz, mov):
        planner = HomotopyThreadingPlanner(0.94, self.grasp_rrt, self.skeletons, goal.skeleton_names)

        planning_t0 = perf_counter()
        sim_grasps = planner.simulate_sampled_grasps(phy, None, viz_execution=False)
        best_grasp = planner.get_best(sim_grasps, viz=viz)
        self.planning_times.append(perf_counter() - planning_t0)
        viz.viz(best_grasp.phy, is_planning=True)
        if best_grasp.res.error_code.val == MoveItErrorCodes.SUCCESS:
            execute_grasp_change_plan(best_grasp, grasp_goal, phy, viz, mov)
            self.traps.reset_trap_detection()
        else:
            print("No plans found!")

    def goal_satisfied(self, goal: ThreadingGoal, phy: Physics):
        if through_skels(self.skeletons, goal.skeleton_names, phy):
            print(f"Through {goal.skeleton_names}!")
            return True
        return False


def execute_grasp_change_plan(best_grasp, grasp_goal, phy, viz, mov):
    deactivate_moving(phy, best_grasp.strategy, viz, is_planning=False, mov=mov)
    pid_to_joint_configs(phy, best_grasp.res, viz, is_planning=False, mov=mov)
    grasp_and_settle(phy, best_grasp.locs, viz, is_planning=False, mov=mov)
    deactivate_release(phy, best_grasp.strategy, viz=viz, is_planning=False, mov=mov)
    grasp_goal.set_grasp_locs(best_grasp.locs)
    print(f"Changed grasp to {best_grasp.locs}")


if __name__ == "__main__":
    main()
