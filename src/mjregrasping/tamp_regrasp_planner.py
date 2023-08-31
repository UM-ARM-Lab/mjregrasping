"""
This baseline is inspired by task and motion planning methods (Not to be confused with TAMPC)

The idea is that to evaluate the cost of a feasibile grasp,
we simply run our MPPI planner for a fixed horizon and take the final/accumulated cost.
"""
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict

import numpy as np

from mjregrasping.goals import ObjectPointGoal
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner
from mjregrasping.regrasp_planner_utils import SimGraspCandidate, SimGraspInput
from mjregrasping.ik import BIG_PENALTY
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.regrasping_mppi import RegraspMPPI, do_grasp_dynamics
from mjregrasping.rollout import control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import val_untangle, Scenario
from mjregrasping.viz import Viz
from moveit_msgs.msg import MoveItErrorCodes


class TAMPRegraspPlanner(HomotopyRegraspPlanner):

    def __init__(self, scenario: Scenario, op_goal: ObjectPointGoal, grasp_rrt: GraspRRT, skeletons: Dict, seed=0):
        super().__init__(op_goal, grasp_rrt, skeletons, seed)
        self.scenario = scenario

    def costs(self, sim_grasp: SimGraspCandidate):
        initial_locs = sim_grasp.initial_locs
        phy_plan = sim_grasp.phy
        candidate_locs = sim_grasp.locs
        res = sim_grasp.res
        strategy = sim_grasp.strategy

        # If there is no significant change in the grasp, that's high cost
        same_locs = abs(initial_locs - candidate_locs) < hp['grasp_loc_diff_thresh']
        not_stay = strategy != Strategies.STAY
        if np.any(same_locs & not_stay):
            cost = 10 * BIG_PENALTY
            return cost

        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            cost = 10 * BIG_PENALTY
            return cost

        return 0, 0

    def simulate_grasp(self, phy: Physics, viz: Optional[Viz], grasp_input: SimGraspInput, viz_execution=False):
        sim_grasp = super().simulate_grasp(phy, viz, grasp_input, viz_execution)

        # now we run the MPPI planner for a fixed horizon and take the final cost
        pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
        mppi = RegraspMPPI(pool=pool, nu=phy.m.nu, seed=0, horizon=hp['horizon'],
                           # FIXME: scenario is hard-coded
                           noise_sigma=self.scenario.noise_sigma, temp=hp['temp'])
        mppi.reset()

        for t in range(10):
            command, sub_time_s = mppi.command(phy, self.op_goal, hp['n_samples'], viz=viz)
            control_step(phy, command, sub_time_s)
            viz.viz(phy, is_planning=True)

            results = op_goal.get_results(phy)
            do_grasp_dynamics(phy, results)

            mppi.roll()

        # TODO: do we need different outputs for the different planners?
        return sim_grasp
