#!/usr/bin/env python3
# NOTE: can't call this file threading.py because it conflicts with the threading module
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Dict, Optional, List

import mujoco
import numpy as np
import rerun as rr
from colorama import Fore

import rospy
from arc_utilities import ros_init
from mjregrasping.goals import GraspLocsGoal, ObjectPointGoal, point_goal_from_geom, ThreadingGoal
from mjregrasping.grasp_and_settle import deactivate_moving, grasp_and_settle, deactivate_release
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_grasp_locs, get_is_grasping
from mjregrasping.homotopy_checker import get_full_h_signature_from_phy, through_skels
from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner
from mjregrasping.homotopy_utils import skeleton_field_dir, NO_HOMOTOPY, make_h_desired, h2array
from mjregrasping.ik import BIG_PENALTY
from mjregrasping.move_to_joint_config import pid_to_joint_configs
from mjregrasping.params import hp
from mjregrasping.physics import Physics, get_q
from mjregrasping.regrasp_planner_utils import SimGraspInput, SimGraspCandidate, get_will_be_grasping
from mjregrasping.regrasping_mppi import RegraspMPPI, mppi_viz
from mjregrasping.rollout import control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import threading
from mjregrasping.teleport_to_plan import teleport_to_end_of_plan
from mjregrasping.trap_detection import TrapDetection
from mjregrasping.trials import load_trial
from mjregrasping.viz import make_viz, Viz
from moveit_msgs.msg import MoveItErrorCodes, MotionPlanResponse
from trajectory_msgs.msg import JointTrajectoryPoint


class HomotopyThreadingPlanner(HomotopyRegraspPlanner):

    def __init__(self, key_loc: float, grasp_rrt: GraspRRT, skeletons: Dict, goal_skel_names: List[str], seed=0):
        super().__init__(key_loc, grasp_rrt, skeletons, seed)
        self.goal_skel_names = goal_skel_names

    def sample_grasp_inputs(self, phy):
        grasps_inputs = []
        is_grasping = get_is_grasping(phy)
        new_is_grasping = 1 - is_grasping
        strategy = [Strategies.NEW_GRASP if is_grasping_i else Strategies.RELEASE for is_grasping_i in new_is_grasping]

        for i in range(hp['threading_n_samples']):
            if i == 0:  # ensure we always try the tip
                sample_loc = 1.0
            else:
                sample_loc = self.sample_loc_with_reflection()
            candidate_locs = []
            for tool_name, s_i in zip(phy.o.rd.tool_sites, strategy):
                if s_i == Strategies.NEW_GRASP:
                    candidate_locs.append(sample_loc)
                elif s_i == Strategies.RELEASE:
                    candidate_locs.append(-1)

            candidate_locs = np.array(candidate_locs)

            grasps_inputs.append(SimGraspInput(strategy, candidate_locs))
        return grasps_inputs

    def sample_loc_with_reflection(self):
        """ Reflection means if we sample a loc like 1.1 which is greater that 1, we reflect that and return 0.9 """
        sample_loc = self.rng.normal(self.key_loc, 0.04)
        if sample_loc > 1.0:
            sample_loc = 2 - sample_loc
        elif sample_loc < 0.0:
            sample_loc = -sample_loc
        return sample_loc

    def is_valid_strategy(self, s, is_grasping):
        is_valid = super().is_valid_strategy(s, is_grasping)
        will_be_grasping = [get_will_be_grasping(s_i, g_i) for s_i, g_i in zip(s, is_grasping)]
        if np.all(will_be_grasping):
            is_valid = False
        if s[0] == Strategies.NEW_GRASP and s[1] == Strategies.MOVE:
            is_valid = False
        if s[1] == Strategies.NEW_GRASP and s[0] == Strategies.MOVE:
            is_valid = False
        return is_valid

    def simulate_grasp(self, phy: Physics, viz: Optional[Viz], grasp_input: SimGraspInput, viz_execution=False):
        strategy = grasp_input.strategy
        candidate_locs = grasp_input.candidate_locs
        initial_locs = get_grasp_locs(phy)

        viz = viz if viz_execution else None
        phy_plan = phy.copy_all()

        deactivate_moving(phy_plan, strategy, viz=viz, is_planning=True)

        # check if we need to move the arms at all
        any_moving = np.any([s in [Strategies.NEW_GRASP, Strategies.MOVE] for s in strategy])
        if any_moving:
            res, scene_msg = self.grasp_rrt.plan(phy_plan, strategy, candidate_locs, viz)

            if res.error_code.val != MoveItErrorCodes.SUCCESS:
                return SimGraspCandidate(phy, phy_plan, strategy, res, candidate_locs, initial_locs)

            if viz_execution and viz is not None:
                self.grasp_rrt.display_result(viz, res, scene_msg)

            # Teleport to the final planned joint configuration
            teleport_to_end_of_plan(phy_plan, res)
            if viz_execution:
                viz.viz(phy_plan, is_planning=True)
        else:
            res = MotionPlanResponse()
            res.error_code.val = MoveItErrorCodes.SUCCESS
            point = JointTrajectoryPoint()
            point.positions = get_q(phy_plan)
            res.trajectory.joint_trajectory.points.append(point)

        grasp_and_settle(phy_plan, candidate_locs, viz, is_planning=True)

        deactivate_release(phy_plan, strategy, viz=viz, is_planning=True)

        return SimGraspCandidate(phy, phy_plan, strategy, res, candidate_locs, initial_locs)

    def costs(self, sim_grasp: SimGraspCandidate):
        costs = super().costs(sim_grasp)

        phy = sim_grasp.phy
        h, _ = get_full_h_signature_from_phy(self.skeletons, phy)
        if h == NO_HOMOTOPY:
            threading_homotopy_cost = BIG_PENALTY
        else:
            h_desired = h2array(make_h_desired(self.skeletons, self.goal_skel_names))
            h = h2array(h)
            threading_homotopy_cost = sum(np.abs(np.array(h) - h_desired)) * BIG_PENALTY

        return costs + (threading_homotopy_cost,)

    def through_skels(self, phy: Physics):
        return through_skels(self.skeletons, self.goal_skel_names, phy)

    def get_cost_names(self):
        cost_names = super().get_cost_names()
        cost_names.append('threading_homotopy')
        return cost_names


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
                disc_center = np.mean(goal.skel[:4], axis=0)
                disc_normal = skeleton_field_dir(goal.skel, disc_center[None])[0] * 0.01
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
