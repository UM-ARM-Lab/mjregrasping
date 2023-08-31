from typing import Dict, Optional

import numpy as np
import rerun as rr
from pymjregrasping_cpp import seedOmpl

from mjregrasping.goal_funcs import get_rope_points, locs_eq
from mjregrasping.goals import ObjectPointGoal
from mjregrasping.grasp_and_settle import release_and_settle, grasp_and_settle
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_grasp_locs, get_is_grasping
from mjregrasping.homotopy_checker import get_full_h_signature_from_phy
from mjregrasping.ik import BIG_PENALTY
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.regrasp_planner_utils import get_geodesic_dist, get_all_strategies_from_phy, SimGraspCandidate, \
    SimGraspInput
from mjregrasping.rrt import GraspRRT
from mjregrasping.teleport_to_plan import teleport_to_end_of_plan
from mjregrasping.viz import Viz
from moveit_msgs.msg import MoveItErrorCodes


class HomotopyRegraspPlanner:

    def __init__(self, op_goal: ObjectPointGoal, grasp_rrt: GraspRRT, skeletons: Dict, seed=0):
        self.op_goal = op_goal
        self.rng = np.random.RandomState(seed)
        self.skeletons = skeletons
        self.true_h_blacklist = []

        self.rrt_rng = np.random.RandomState(seed)
        self.grasp_rrt = grasp_rrt
        seedOmpl(seed)

    def update_blacklists(self, phy):
        current_true_h, _ = get_full_h_signature_from_phy(self.skeletons, phy)
        new = True
        for blacklisted_true_h in self.true_h_blacklist:
            if current_true_h == blacklisted_true_h:
                new = False
                break
        if new:
            self.true_h_blacklist.append(current_true_h)

    def simulate_sampled_grasps(self, phy, viz, viz_execution=False):
        grasps_inputs = self.sample_grasp_inputs(phy)
        sim_grasps = self.simulate_grasps(grasps_inputs, phy, viz, viz_execution)
        return sim_grasps

    def simulate_grasps(self, grasps_inputs, phy, viz, viz_execution):
        sim_grasps = []
        for grasp_input in grasps_inputs:
            sim_grasp = self.simulate_grasp(phy, viz, grasp_input, viz_execution)
            sim_grasps.append(sim_grasp)
        return sim_grasps

    def sample_grasp_inputs(self, phy):
        grasps_inputs = []
        is_grasping = get_is_grasping(phy)
        for strategy in get_all_strategies_from_phy(phy):
            for i in range(hp['n_grasp_samples']):
                if i == 0:
                    sample_loc = self.op_goal.loc
                elif i == 1:
                    sample_loc = 0
                elif i == 2:
                    sample_loc = 1
                else:
                    sample_loc = self.rng.uniform(0, 1)
                candidate_locs = []
                for tool_name, s_i, is_grasping_i in zip(phy.o.rd.tool_sites, strategy, is_grasping):
                    if s_i in [Strategies.NEW_GRASP, Strategies.MOVE]:
                        candidate_locs.append(sample_loc)
                    elif s_i in [Strategies.RELEASE, Strategies.STAY]:
                        candidate_locs.append(-1)

                candidate_locs = np.array(candidate_locs)

                grasps_inputs.append(SimGraspInput(strategy, candidate_locs))
        return grasps_inputs

    def get_best(self, sim_grasps, viz: Optional[Viz]) -> SimGraspCandidate:
        costs_lists = []
        total_costs = []
        for sim_grasp in sim_grasps:
            costs = self.costs(sim_grasp)
            cost_i = sum(costs)
            costs_lists.append(costs)
            total_costs.append(cost_i)

        sorted_idxs = np.argsort(total_costs)
        sorted_cost_lists = [costs_lists[i] for i in sorted_idxs]
        sorted_grasps = [sim_grasps[i] for i in sorted_idxs]
        best_sim_grasp = sorted_grasps[0]

        if not viz:
            return best_sim_grasp

        cost_names = self.get_cost_names()
        for i, (sim_grasp, costs_list) in enumerate(zip(sorted_grasps, sorted_cost_lists)):
            rr.set_time_sequence('regrasp_planner', i)
            viz.viz(sim_grasp.phy, is_planning=True)
            cost_i = sum(costs_list)
            rr.log_tensor('homotopy_costs', [BIG_PENALTY, cost_i] + list(costs_list))
            msg = " ".join([f"{name}: {cost:.2f}" for name, cost in zip(cost_names, costs_list)])
            rr.log_tensor('homotopy_costs', msg)

        return best_sim_grasp

    def costs(self, sim_grasp: SimGraspCandidate):
        phy0 = sim_grasp.phy0
        initial_locs = sim_grasp.initial_locs
        phy_plan = sim_grasp.phy
        candidate_locs = sim_grasp.locs
        res = sim_grasp.res
        strategy = sim_grasp.strategy

        # If there is no significant change in the grasp, that's high cost
        same_locs = locs_eq(initial_locs, candidate_locs)
        not_stay = strategy != Strategies.STAY
        if np.any(same_locs & not_stay):
            cost = 10 * BIG_PENALTY
            return cost, 0, 0, 0, 0

        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            cost = 10 * BIG_PENALTY
            return cost, 0, 0, 0, 0

        homotopy_cost = 0
        h_plan, loops_plan = get_full_h_signature_from_phy(self.skeletons, phy_plan)
        rope_points_plan = get_rope_points(phy_plan)
        for blacklisted_h in self.true_h_blacklist:
            if h_plan == blacklisted_h:
                homotopy_cost = BIG_PENALTY
                break

        geodesics_cost = get_geodesic_dist(candidate_locs, self.op_goal) * hp['geodesic_weight']

        prev_plan_pos = res.trajectory.joint_trajectory.points[0].positions
        dq = 0
        for point in res.trajectory.joint_trajectory.points[1:]:
            plan_pos = point.positions
            dq += np.linalg.norm(np.array(plan_pos) - np.array(prev_plan_pos))
        dq_cost = np.clip(dq, 0, BIG_PENALTY / 2) * hp['robot_dq_weight']

        rope_points0 = get_rope_points(phy0)
        drope = np.linalg.norm(rope_points_plan - rope_points0, axis=-1).mean()
        drope_cost = np.clip(drope, 0, BIG_PENALTY / 2) * hp['rope_dq_weight']

        return 0, dq_cost, drope_cost, homotopy_cost, geodesics_cost

    def get_cost_names(self):
        return ["penalty", "dq", "drope", "homotopy", "geodesic"]

    def cost(self, sim_grasp: SimGraspCandidate):
        costs = self.costs(sim_grasp)
        return sum(costs)

    def simulate_grasp(self, phy: Physics, viz: Optional[Viz], grasp_input: SimGraspInput, viz_execution=False):
        strategy = grasp_input.strategy
        candidate_locs = grasp_input.candidate_locs
        initial_locs = get_grasp_locs(phy)

        viz = viz if viz_execution else None
        phy_plan = phy.copy_all()

        # release and settle for any moving grippers
        release_and_settle(phy_plan, strategy, viz=viz, is_planning=True)

        res, scene_msg = self.grasp_rrt.plan(phy_plan, strategy, candidate_locs, viz)

        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            return SimGraspCandidate(phy, phy_plan, strategy, res, candidate_locs, initial_locs)
        print(f"plan has {len(res.trajectory.joint_trajectory.points)} points")

        if viz_execution and viz is not None:
            self.grasp_rrt.display_result(viz, res, scene_msg)

        # Teleport to the final planned joint configuration
        teleport_to_end_of_plan(phy_plan, res)
        if viz_execution:
            viz.viz(phy_plan, is_planning=True)

        # Activate grasps
        grasp_and_settle(phy_plan, candidate_locs, viz, is_planning=True)

        return SimGraspCandidate(phy, phy_plan, strategy, res, candidate_locs, initial_locs)
