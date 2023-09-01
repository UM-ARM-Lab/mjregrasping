"""
This baseline is inspired by TAMPC

 - traps are points in 3D, with no action, corresponding to the location of the grasped points on the rope.
 - trap cost is thus the total distance between the candidate grasp locations and the trap locations
"""
from typing import Dict

import numpy as np

from mjregrasping.geometry import pairwise_squared_distances
from mjregrasping.goals import ObjectPointGoal
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_grasp_locs
from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner
from mjregrasping.regrasp_planner_utils import get_geodesic_dist, SimGraspCandidate
from mjregrasping.ik import BIG_PENALTY
from mjregrasping.params import hp
from mjregrasping.rrt import GraspRRT
from moveit_msgs.msg import MoveItErrorCodes


class ExploreLocsRegraspPlanner(HomotopyRegraspPlanner):

    def __init__(self, key_loc: float, grasp_rrt: GraspRRT, skeletons: Dict, seed=0):
        super().__init__(key_loc, grasp_rrt, skeletons, seed)
        self.trap_locs = []

    def update_blacklists(self, phy):
        locs = get_grasp_locs(phy)
        for loc_i in locs:
            if loc_i != -1:
                self.trap_locs.append(loc_i)

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
            return 10 * BIG_PENALTY, 0, 0, 0

        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            return 10 * BIG_PENALTY, 0, 0, 0

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
            # trap_xpos = grasp_locations_to_xpos(phy_plan, self.trap_locs)
            # if viz:
            #     viz.points('traps', trap_xpos, color='r', radius=0.04)

        geodesics_cost = get_geodesic_dist(candidate_locs, self.key_loc) * hp['geodesic_weight']

        prev_plan_pos = res.trajectory.joint_trajectory.points[0].positions
        dq = 0
        for point in res.trajectory.joint_trajectory.points[1:]:
            plan_pos = point.positions
            dq += np.linalg.norm(np.array(plan_pos) - np.array(prev_plan_pos))
        dq_cost = np.clip(dq, 0, BIG_PENALTY / 2) * hp['robot_dq_weight']

        return 0, dq_cost, trap_cost, geodesics_cost

    def get_cost_names(self):
        return ['penalty', 'dq', 'trap', 'geodesics']
