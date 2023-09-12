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
from mjregrasping.geometry import pairwise_squared_distances
from mjregrasping.goal_funcs import get_tool_points
from mjregrasping.goals import GraspLocsGoal, ObjectPointGoal, point_goal_from_geom, ThreadingGoal
from mjregrasping.grasp_and_settle import deactivate_moving, grasp_and_settle, deactivate_release
from mjregrasping.grasp_conversions import grasp_locations_to_xpos
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
from mjregrasping.regrasping_mppi import RegraspMPPI, mppi_viz, do_grasp_dynamics
from mjregrasping.rollout import control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import threading_cable, Scenario
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
        if hp['use_signature_cost']:
            if h == NO_HOMOTOPY:
                threading_signature_cost = BIG_PENALTY
            else:
                h_desired = h2array(make_h_desired(self.skeletons, self.goal_skel_names))
                h = h2array(h)
                threading_signature_cost = sum(np.abs(np.array(h) - h_desired)) * BIG_PENALTY
        else:
            threading_signature_cost = 0

        return costs + (threading_signature_cost,)

    def get_cost_names(self):
        cost_names = super().get_cost_names()
        cost_names.append('threading_homotopy')
        return cost_names


class TAMPThreadingPlanner(HomotopyThreadingPlanner):

    def __init__(self, scenario: Scenario, key_loc: float, grasp_rrt: GraspRRT, skeletons: Dict,
                 next_goal: ThreadingGoal, seed=0):
        super().__init__(key_loc, grasp_rrt, skeletons, seed)
        self.scenario = scenario
        self.next_goal = next_goal

    def simulate_grasp(self, phy: Physics, viz: Optional[Viz], grasp_input: SimGraspInput, viz_execution=False):
        sim_grasp = super().simulate_grasp(phy, viz, grasp_input, viz_execution)
        if sim_grasp.res.error_code.val != MoveItErrorCodes.SUCCESS:
            sim_grasp.phy1 = sim_grasp.phy.copy_data()
            return sim_grasp

        phy_plan = sim_grasp.phy.copy_data()

        self.next_goal.grasp_goal.set_grasp_locs(sim_grasp.locs, is_planning=True)

        # now we run the MPPI planner for a fixed horizon and take the final cost
        pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
        mppi = RegraspMPPI(pool=pool, nu=phy_plan.m.nu, seed=0, horizon=hp['horizon'],
                           noise_sigma=self.scenario.noise_sigma, temp=hp['temp'])
        mppi.reset()

        for t in range(hp['tamp_horizon']):
            command, sub_time_s = mppi.command(phy_plan, self.next_goal, hp['n_samples'], viz=viz)
            control_step(phy_plan, command, sub_time_s)
            if viz and viz_execution:
                viz.viz(phy_plan, is_planning=True)

            do_grasp_dynamics(phy_plan)

            mppi.roll()

        sim_grasp.phy1 = phy_plan
        return sim_grasp

    def costs(self, sim_grasp: SimGraspCandidate):
        # these costs are computed using the phy after the grasp but before the post-grasp motion towards the goal
        # NOTE: even though the homotopy cost is computed, because we never blacklist anything in this baseline, it's always 0
        costs = HomotopyRegraspPlanner.costs(self, sim_grasp)

        post_motion_phy = sim_grasp.phy1
        keypoint = grasp_locations_to_xpos(post_motion_phy, [self.next_goal.loc])[0]
        final_keypoint_dist = self.next_goal.keypoint_dist_to_goal(keypoint) * hp['final_keypoint_dist_weight']

        return costs + (final_keypoint_dist,)

    def get_cost_names(self):
        return HomotopyRegraspPlanner.get_cost_names(self) + ['final_keypoint_dist']


@ros_init.with_ros("threading")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('threading')
    rr.connect()

    scenario = threading_cable
    hp["threading_n_samples"] = 10

    viz = make_viz(scenario)

    gl_ctx = mujoco.GLContext(1280, 720)
    gl_ctx.make_current()

    grasp_rrt = GraspRRT()

    for i in range(0, 25):
        phy, sdf, skeletons, mov = load_trial(i, gl_ctx, scenario, viz)

        grasp_goal = GraspLocsGoal(get_grasp_locs(phy))

        pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
        mppi = RegraspMPPI(pool=pool, nu=phy.m.nu, seed=i, horizon=hp['horizon'],
                           noise_sigma=scenario.noise_sigma,
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

        # method = ThreadingMethodOurs(grasp_rrt, skeletons, traps, end_loc)
        method = ThreadingMethodWang(grasp_rrt, skeletons, traps, end_loc)
        # method = ThreadingMethodTAMP(scenario, grasp_rrt, skeletons, traps, end_loc)
        print(f"Running method {method.__class__.__name__}")

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
                    success = through_skels(skeletons, goals[-2].skeleton_names, phy)
                    print(Fore.GREEN + f"Task complete! {success=}" + Fore.RESET)
                    break
            else:
                disc_center = np.mean(goal.skel[:4], axis=0)
                disc_normal = skeleton_field_dir(goal.skel, disc_center[None])[0] * 0.01
                disc_rad = 0.05  # TODO: compute the radius of the disc

                disc_penetrated = goal.satisfied(phy, disc_center, disc_normal, disc_rad)
                is_stuck = traps.check_is_stuck(phy, grasp_goal)
                if disc_penetrated:
                    print("Disc penetrated!")
                    mppi.reset()
                    method.on_disc(phy, goal, goals[goal_idx + 1], viz, mov)
                elif is_stuck:
                    print("Stuck!")
                    mppi.reset()
                    method.on_stuck(phy, goal, viz, mov)

                if method.goal_satisfied(goal, phy):
                    goal_idx += 1
                    goal = goals[goal_idx]

            mpc_t0 = perf_counter()
            command, sub_time_s = mppi.command(phy, goal, num_samples, viz=viz)
            mpc_times.append(perf_counter() - mpc_t0)
            mppi_viz(mppi, goal, phy, command, sub_time_s)

            control_step(phy, command, sub_time_s, mov=mov)
            viz.viz(phy)

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
            'method':         method.method_name(),
            'hp':             hp,
        }
        mov.close(metrics)


class ThreadingMethod:

    def __init__(self, grasp_rrt: GraspRRT, skeletons: Dict, traps: TrapDetection, end_loc: float):
        self.grasp_rrt = grasp_rrt
        self.skeletons = skeletons
        self.traps = traps
        self.end_loc = end_loc
        self.planning_times = []

    def method_name(self):
        raise NotImplementedError()

    def on_disc(self, phy, goal, next_goal, viz, mov):
        raise NotImplementedError()

    def on_stuck(self, phy, goal, viz, mov):
        raise NotImplementedError()

    def goal_satisfied(self, goal, phy):
        raise NotImplementedError()

    def get_scootched_loc(self, grasp_goal: GraspLocsGoal):
        # If the robot isn't grasping anything, grab near the end but with enough room to stick it through the loop
        # and grasp the other end
        if np.all(grasp_goal.get_grasp_locs() == -1):
            loc = self.end_loc - hp['grasp_loc_diff_thresh'] * 2
        else:
            loc = max(grasp_goal.get_grasp_locs()) - hp['grasp_loc_diff_thresh'] * 2
        return loc

    def scootch_goal(self, phy: Physics, goal: ThreadingGoal):
        goal.loc = max(goal.loc - hp['goal_scootch'], max(get_grasp_locs(phy)))


class ThreadingMethodWang(ThreadingMethod):
    """
    Based on "An Online Method for Tight-tolerance Insertion Tasks for String and Rope"
    by Wang, Berenson, and Balkcom. 2015
    """

    def __init__(self, grasp_rrt: GraspRRT, skeletons: Dict, traps: TrapDetection, end_loc: float):
        super().__init__(grasp_rrt, skeletons, traps, end_loc)
        self.plan_to_end_found = False

    def method_name(self):
        return "Wang et al."

    def on_disc(self, phy, goal, next_goal, viz, mov):
        self.plan_to_end_found = False

        # The original paper uses floating grippers that can teleport, so adapt that by alterating grippers
        locs, strategy = self.get_strategy_and_locs(phy, self.end_loc)
        planning_t0 = perf_counter()
        self.grasp_rrt.fix_start_state_in_place(phy)
        res, scene_msg = self.grasp_rrt.plan(phy, strategy, locs, viz)
        self.planning_times.append(perf_counter() - planning_t0)
        if res.error_code.val == MoveItErrorCodes.SUCCESS:
            grasp = SimGraspCandidate(None, None, strategy, res, locs, None)
            execute_grasp_change_plan(grasp, goal.grasp_goal, phy, viz, mov)
            self.traps.reset_trap_detection()
            self.plan_to_end_found = True
            return
        else:
            self.scootch_goal(phy, goal)
        print("No plans found!")

    def on_stuck(self, phy, goal, viz, mov):
        # In Wang et. al, they only check whether the gripper is too close to the obstacle, not the more general
        # case of the controller being stuck, so we do that here.
        tool_points = get_tool_points(phy)
        gripper_to_goal_dist = np.min(np.sqrt(pairwise_squared_distances(tool_points, goal.goal_point[None])))
        if gripper_to_goal_dist > 0.05:
            return

        # Try to scootch down
        loc = self.get_scootched_loc(goal.grasp_goal)
        locs, strategy = self.get_strategy_and_locs(phy, loc)
        planning_t0 = perf_counter()
        self.grasp_rrt.fix_start_state_in_place(phy)
        res, scene_msg = self.grasp_rrt.plan(phy, strategy, locs, viz)
        self.planning_times.append(perf_counter() - planning_t0)
        if res.error_code.val == MoveItErrorCodes.SUCCESS:
            print("Scootched down!")
            grasp = SimGraspCandidate(None, None, strategy, res, locs, None)
            execute_grasp_change_plan(grasp, goal.grasp_goal, phy, viz, mov)
            self.traps.reset_trap_detection()
            return
        else:
            self.scootch_goal(phy, goal)
        print("No plans found!")

    def goal_satisfied(self, goal, phy):
        ret = self.plan_to_end_found
        self.plan_to_end_found = False
        return ret

    def get_strategy_and_locs(self, phy, loc):
        strategy = []
        locs = []
        for g_i in get_is_grasping(phy):
            if g_i:
                strategy.append(Strategies.RELEASE)
                locs.append(-1)
            else:
                strategy.append(Strategies.NEW_GRASP)
                locs.append(loc)
        locs = np.array(locs)
        return locs, strategy


class ThreadingMethodTAMP(ThreadingMethod):

    def __init__(self, scenario: Scenario, grasp_rrt: GraspRRT, skeletons: Dict, traps: TrapDetection, end_loc: float):
        super().__init__(grasp_rrt, skeletons, traps, end_loc)
        self.rng = np.random.RandomState(0)
        self.plan_to_end_found = False
        self.scenario = scenario

    def on_disc(self, phy, goal, next_goal, viz, mov):
        planner = TAMPThreadingPlanner(self.scenario, self.end_loc, self.grasp_rrt, self.skeletons, next_goal)
        print(f"Planning with {planner.key_loc=}...")
        planning_t0 = perf_counter()
        sim_grasps = planner.simulate_sampled_grasps(phy, viz, viz_execution=True)
        best_grasp = planner.get_best(sim_grasps, viz=viz)
        self.planning_times.append(perf_counter() - planning_t0)
        viz.viz(best_grasp.phy1, is_planning=True)
        if best_grasp.res.error_code.val == MoveItErrorCodes.SUCCESS:
            print("Executing grasp change plan")
            self.plan_to_end_found = True
            execute_grasp_change_plan(best_grasp, goal.grasp_goal, phy, viz, mov)
            self.traps.reset_trap_detection()
        else:
            # if we've reached the goal but can't grasp the end, scootch down the rope
            # but don't scootch past where we are currently grasping it.
            self.scootch_goal(phy, goal)
            print("No plans found!")
            self.plan_to_end_found = False

    def on_stuck(self, phy, goal, viz, mov):
        loc = self.get_scootched_loc(goal.grasp_goal)
        planner = TAMPThreadingPlanner(self.scenario, loc, self.grasp_rrt, self.skeletons, goal)

        planning_t0 = perf_counter()
        sim_grasps = planner.simulate_sampled_grasps(phy, None, viz_execution=False)
        best_grasp = planner.get_best(sim_grasps, viz=viz)
        self.planning_times.append(perf_counter() - planning_t0)
        viz.viz(best_grasp.phy, is_planning=True)
        if best_grasp.res.error_code.val == MoveItErrorCodes.SUCCESS:
            execute_grasp_change_plan(best_grasp, goal.grasp_goal, phy, viz, mov)
            self.traps.reset_trap_detection()
        else:
            self.scootch_goal(phy, goal)
            print("No plans found!")

    def goal_satisfied(self, goal: ThreadingGoal, phy: Physics):
        ret = self.plan_to_end_found
        self.plan_to_end_found = False
        return ret

    def method_name(self):
        return f"Tamp{hp['tamp_horizon']}'"


class ThreadingMethodOurs(ThreadingMethod):

    def on_disc(self, phy, goal, next_goal, viz, mov):
        planner = HomotopyThreadingPlanner(self.end_loc, self.grasp_rrt, self.skeletons, goal.skeleton_names)
        print(f"Planning with {planner.key_loc=}...")
        planning_t0 = perf_counter()
        sim_grasps = planner.simulate_sampled_grasps(phy, None, viz_execution=False)
        best_grasp = planner.get_best(sim_grasps, viz=viz)
        self.planning_times.append(perf_counter() - planning_t0)
        viz.viz(best_grasp.phy, is_planning=True)
        if best_grasp.res.error_code.val == MoveItErrorCodes.SUCCESS:
            if through_skels(self.skeletons, goal.goal_skel_names, best_grasp.phy):
                print("Executing grasp change plan")
                execute_grasp_change_plan(best_grasp, goal.grasp_goal, phy, viz, mov)
                self.traps.reset_trap_detection()
            else:
                print("Not through the goal skeleton!")
        else:
            # if we've reached the goal but can't grasp the end, scootch down the rope
            # but don't scootch past where we are currently grasping it.
            self.scootch_goal(phy, goal)
            print("No plans found!")

    def on_stuck(self, phy, goal, viz, mov):
        loc = self.get_scootched_loc(goal.grasp_goal)
        planner = HomotopyThreadingPlanner(loc, self.grasp_rrt, self.skeletons, goal.skeleton_names)

        planning_t0 = perf_counter()
        sim_grasps = planner.simulate_sampled_grasps(phy, None, viz_execution=False)
        best_grasp = planner.get_best(sim_grasps, viz=viz)
        self.planning_times.append(perf_counter() - planning_t0)
        viz.viz(best_grasp.phy, is_planning=True)
        if best_grasp.res.error_code.val == MoveItErrorCodes.SUCCESS:
            execute_grasp_change_plan(best_grasp, goal.grasp_goal, phy, viz, mov)
            self.traps.reset_trap_detection()
        else:
            self.scootch_goal(phy, goal)
            print("No plans found!")

    def goal_satisfied(self, goal: ThreadingGoal, phy: Physics):
        if through_skels(self.skeletons, goal.skeleton_names, phy):
            print(f"Through {goal.skeleton_names}!")
            return True
        return False

    def method_name(self):
        return "\\signature{}"


def execute_grasp_change_plan(best_grasp, grasp_goal: GraspLocsGoal, phy: Physics, viz: Viz, mov):
    deactivate_moving(phy, best_grasp.strategy, viz, is_planning=False, mov=mov)
    pid_to_joint_configs(phy, best_grasp.res, viz, is_planning=False, mov=mov)
    grasp_and_settle(phy, best_grasp.locs, viz, is_planning=False, mov=mov)
    deactivate_release(phy, best_grasp.strategy, viz=viz, is_planning=False, mov=mov)
    grasp_goal.set_grasp_locs(best_grasp.locs)
    print(f"Changed grasp to {best_grasp.locs}")


if __name__ == "__main__":
    main()
