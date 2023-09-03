#!/usr/bin/env python3
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
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
from mjregrasping.move_to_joint_config import execute_grasp_plan
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

        for i in range(6):
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

    viz = make_viz(scenario)

    gl_ctx = mujoco.GLContext(1280, 720)
    gl_ctx.make_current()

    grasp_rrt = GraspRRT()

    for i in range(10):
        phy, sdf, skeletons, mov = load_trial(i, gl_ctx, scenario, viz)

        grasp_goal = GraspLocsGoal(get_grasp_locs(phy))

        pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
        mppi = RegraspMPPI(pool=pool, nu=phy.m.nu, seed=i, horizon=hp['horizon'],
                           noise_sigma=threading.noise_sigma,
                           temp=hp['temp'])
        num_samples = hp['n_samples']
        goal_idx = 0

        mppi.reset()

        end_loc = 1.0
        goals = [
            ThreadingGoal(grasp_goal, skeletons, ['loop1'], end_loc, viz),
            ThreadingGoal(grasp_goal, skeletons, ['loop1', 'loop2'], end_loc, viz),
            ThreadingGoal(grasp_goal, skeletons, ['loop1', 'loop2', 'loop3'], end_loc, viz),
            point_goal_from_geom(grasp_goal, phy, "goal", 1, viz)
        ]
        goal = goals[goal_idx]

        traps = TrapDetection()

        itr = 0

        viz.viz(phy)
        success = False
        while True:
            if rospy.is_shutdown():
                mov.close()
                break

            if itr > 250:
                print(Fore.RED + "Max iterations reached!" + Fore.RESET)
                break

            goal.viz_goal(phy)

            if isinstance(goal, ObjectPointGoal):
                if goal.satisfied(phy):
                    print(Fore.GREEN + "Task complete!" + Fore.RESET)
                    break
            else:
                disc_center = np.mean(goal.skel[:4], axis=0)
                disc_normal = skeleton_field_dir(goal.skel, disc_center[None])[0] * 0.01
                disc_rad = 0.05  # TODO: compute the radius of the disc

                disc_penetrated = goal.satisfied(phy, disc_center, disc_normal, disc_rad)
                is_stuck = traps.check_is_stuck(phy)
                if disc_penetrated:
                    goal.loc -= hp['scootch_fraction']

                    mppi.reset()

                    planner = HomotopyThreadingPlanner(end_loc, grasp_rrt, skeletons, goal.skeleton_names)

                    print(f"Planning with {planner.key_loc=}...")
                    sim_grasps = planner.simulate_sampled_grasps(phy, None, viz_execution=False)
                    best_grasp = planner.get_best(sim_grasps, viz=viz)
                    viz.viz(best_grasp.phy, is_planning=True)
                    if best_grasp.res.error_code.val == MoveItErrorCodes.SUCCESS:
                        if planner.through_skels(best_grasp.phy):
                            print("Executing grasp change plan")
                            execute_grasp_change_plan(best_grasp, grasp_goal, phy, viz, mov)

                            traps.reset_trap_detection()
                        else:
                            print("Not through the goal skeleton!")
                    else:
                        print("No plans found!")
                elif is_stuck:
                    mppi.reset()

                    planner = HomotopyThreadingPlanner(0.94, grasp_rrt, skeletons, goal.skeleton_names)

                    sim_grasps = planner.simulate_sampled_grasps(phy, None, viz_execution=False)
                    best_grasp = planner.get_best(sim_grasps, viz=viz)
                    viz.viz(best_grasp.phy, is_planning=True)
                    if best_grasp.res.error_code.val == MoveItErrorCodes.SUCCESS:
                        execute_grasp_change_plan(best_grasp, grasp_goal, phy, viz, mov)
                        traps.reset_trap_detection()
                    else:
                        print("No plans found!")

                if through_skels(skeletons, goal.skeleton_names, phy):
                    print(f"Through {goal.skeleton_names}!")
                    goal_idx += 1
                    goal = goals[goal_idx]

            command, sub_time_s = mppi.command(phy, goal, num_samples, viz=viz)
            mppi_viz(mppi, goal, phy, command, sub_time_s)

            control_step(phy, command, sub_time_s, mov=mov)
            viz.viz(phy)

            # do_grasp_dynamics(phy)

            mppi.roll()

            itr += 1

        # save the results
        metrics = {
            'itr':     itr,
            'success': success,
            'time':    phy.d.time
        }
        mov.close(metrics)


def execute_grasp_change_plan(best_grasp, grasp_goal, phy, viz, mov):
    deactivate_moving(phy, best_grasp.strategy, viz, is_planning=False, mov=mov)
    execute_grasp_plan(phy, best_grasp.res, viz, is_planning=False, mov=mov)
    grasp_and_settle(phy, best_grasp.locs, viz, is_planning=False, mov=mov)
    deactivate_release(phy, best_grasp.strategy, viz=viz, is_planning=False, mov=mov)
    grasp_goal.set_grasp_locs(best_grasp.locs)


if __name__ == "__main__":
    main()
