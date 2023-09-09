"""
This baseline is inspired by task and motion planning methods (Not to be confused with TAMPC)

The idea is that to evaluate the cost of a feasibile grasp,
we simply run our MPPI planner for a fixed horizon and take the final/accumulated cost.
"""
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from typing import Optional, Dict

from mjregrasping.goals import ObjectPointGoal
from mjregrasping.grasp_conversions import grasp_locations_to_xpos
from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.regrasp_planner_utils import SimGraspCandidate, SimGraspInput
from mjregrasping.regrasping_mppi import RegraspMPPI, do_grasp_dynamics
from mjregrasping.rollout import control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import Scenario
from mjregrasping.viz import Viz
from moveit_msgs.msg import MoveItErrorCodes


class TAMPRegraspPlanner(HomotopyRegraspPlanner):

    def __init__(self, scenario: Scenario, goal: ObjectPointGoal, grasp_rrt: GraspRRT, skeletons: Dict, seed=0):
        self.goal = copy(goal)
        super().__init__(goal.loc, grasp_rrt, skeletons, seed)
        self.scenario = scenario

    def costs(self, sim_grasp: SimGraspCandidate):
        # these costs are computed using the phy after the grasp but before the post-grasp motion towards the goal
        # NOTE: even though the homotopy cost is computed, because we never blacklist anything in this baseline, it's always 0
        costs = super().costs(sim_grasp)

        post_motion_phy = sim_grasp.phy1
        keypoint = grasp_locations_to_xpos(post_motion_phy, [self.goal.loc])[0]
        final_keypoint_dist = self.goal.keypoint_dist_to_goal(keypoint) * hp['final_keypoint_dist_weight']

        return costs + (final_keypoint_dist,)

    def get_cost_names(self):
        return super().get_cost_names() + ['final_keypoint_dist']

    def simulate_grasp(self, phy: Physics, viz: Optional[Viz], grasp_input: SimGraspInput, viz_execution=False):
        sim_grasp = super().simulate_grasp(phy, viz, grasp_input, viz_execution)
        if sim_grasp.res.error_code.val != MoveItErrorCodes.SUCCESS:
            sim_grasp.phy1 = sim_grasp.phy.copy_data()
            return sim_grasp

        phy_plan = sim_grasp.phy.copy_data()

        self.goal.grasp_goal.set_grasp_locs(sim_grasp.locs, is_planning=True)

        # now we run the MPPI planner for a fixed horizon and take the final cost
        pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
        mppi = RegraspMPPI(pool=pool, nu=phy_plan.m.nu, seed=0, horizon=hp['horizon'],
                           noise_sigma=self.scenario.noise_sigma, temp=hp['temp'])
        mppi.reset()

        for t in range(hp['tamp_horizon']):
            command, sub_time_s = mppi.command(phy_plan, self.goal, hp['n_samples'], viz=viz)
            control_step(phy_plan, command, sub_time_s)
            if viz and viz_execution:
                viz.viz(phy_plan, is_planning=True)

            do_grasp_dynamics(phy_plan)

            mppi.roll()

        sim_grasp.phy1 = phy_plan
        return sim_grasp
