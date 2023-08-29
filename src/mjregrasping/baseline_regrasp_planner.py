"""
This baseline is inspired by TAMPC

 - traps are points in 3D, with no action, corresponding to the location of the grasped points on the rope.
 - trap cost is thus the total distance between the candidate grasp locations and the trap locations
"""
from typing import Dict, Optional

import numpy as np

from mjregrasping.geometry import pairwise_squared_distances
from mjregrasping.goals import ObjectPointGoal
from mjregrasping.grasp_conversions import grasp_locations_to_xpos
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_grasp_locs
from mjregrasping.homotopy_checker import CollisionChecker, AllowablePenetration
from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner, SimGraspCandidate, rr_log_costs, \
    get_geodesic_dist
from mjregrasping.ik import BIG_PENALTY
from mjregrasping.params import hp
from mjregrasping.rrt import GraspRRT
from mjregrasping.viz import Viz
from moveit_msgs.msg import MoveItErrorCodes


class BaselineRegraspPlanner(HomotopyRegraspPlanner):

    def __init__(self, op_goal: ObjectPointGoal, grasp_rrt: GraspRRT, skeletons: Dict, seed=0):
        super().__init__(op_goal, grasp_rrt, skeletons, seed)
        self.trap_locs = []

    def update_blacklists(self, phy):
        locs = get_grasp_locs(phy)
        for loc_i in locs:
            if loc_i != -1:
                self.trap_locs.append(loc_i)

    def cost(self, sim_grasp: SimGraspCandidate, viz: Optional[Viz], log_loops=False):
        initial_locs = sim_grasp.initial_locs
        phy_plan = sim_grasp.phy
        candidate_locs = sim_grasp.locs
        candidate_dxpos = sim_grasp.candidate_dxpos
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

        # for the candidate grasp locations, get the corresponding xpos
        # then compute the distance matrix between the candidate xposes and the trap xposes
        if len(self.trap_locs) == 0:
            trap_cost = 0
        else:
            valid_candidate_locs = np.array(list(filter(lambda x: x != -1, candidate_locs)))
            trap_locs_2d = np.atleast_2d(self.trap_locs)
            valid_candidate_locs_2d = np.atleast_2d(valid_candidate_locs)
            dists = pairwise_squared_distances(valid_candidate_locs_2d, trap_locs_2d)
            mean_dists = np.mean(dists, axis=1)
            trap_cost = np.sum(np.exp(-mean_dists) * hp['trap_weight'])
            trap_xpos = grasp_locations_to_xpos(phy_plan, self.trap_locs)
            if viz:
                viz.points('traps', trap_xpos, color='r', radius=0.04)

        geodesics_cost = get_geodesic_dist(candidate_locs, self.op_goal) * hp['geodesic_weight']

        prev_plan_pos = res.trajectory.joint_trajectory.points[0].positions
        dq = 0
        for point in res.trajectory.joint_trajectory.points[1:]:
            plan_pos = point.positions
            dq += np.linalg.norm(np.array(plan_pos) - np.array(prev_plan_pos))
        dq_cost = np.clip(dq, 0, BIG_PENALTY / 2) * hp['robot_dq_weight']

        cost = sum([
            dq_cost,
            trap_cost,
            geodesics_cost,
        ])

        rr_log_costs(entity_path='regrasp_costs', entity_paths=[
            'regrasp_costs/dq_cost',
            'regrasp_costs/trap_cost',
            'regrasp_costs/geodesics_cost',
        ], values=[
            dq_cost,
            trap_cost,
            geodesics_cost,
        ], colors=[
            [0.5, 0.5, 0, 1.0],
            [0, 1, 0, 1.0],
            [1, 0, 0, 1.0],
        ])

        return cost
