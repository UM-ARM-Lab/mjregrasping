from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
from pymjregrasping_cpp import seedOmpl

from mjregrasping.goal_funcs import get_rope_points, locs_eq
from mjregrasping.grasp_and_settle import deactivate_release_and_moving, grasp_and_settle
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_grasp_locs, get_is_grasping
from mjregrasping.homotopy_checker import get_full_h_signature_from_phy
from mjregrasping.homotopy_utils import NO_HOMOTOPY, h2array, make_h_desired
from mjregrasping.ik import BIG_PENALTY
from mjregrasping.params import hp
from mjregrasping.physics import Physics, get_q
from mjregrasping.regrasp_planner_utils import get_geodesic_dist, get_all_strategies_from_phy, SimGraspCandidate, \
    SimGraspInput, get_will_be_grasping
from mjregrasping.rrt import GraspRRT
from mjregrasping.teleport_to_plan import teleport_to_end_of_plan
from mjregrasping.viz import Viz, plt_fig_to_img_np
from moveit_msgs.msg import MoveItErrorCodes, MotionPlanResponse
from trajectory_msgs.msg import JointTrajectoryPoint


def get_robot_dq_cost(res):
    prev_plan_pos = res.trajectory.joint_trajectory.points[0].positions
    dq = 0
    for point in res.trajectory.joint_trajectory.points[1:]:
        plan_pos = point.positions
        dq += np.sum(np.square(np.array(plan_pos) - np.array(prev_plan_pos)))
    dq_cost = np.clip(dq * hp['robot_dq_weight1'], 0, BIG_PENALTY * hp['robot_dq_weight2'])
    return dq_cost


class HomotopyRegraspPlanner:

    def __init__(self, key_loc: float, grasp_rrt: GraspRRT, skeletons: Dict,
                 goal_skel_names: Optional[List[str]] = None, seed=0):
        """

        Args:
            key_loc: The location on the rope which we care about "using" for the task. The cost will try to minimize
                     geometric distance to this location.
            grasp_rrt:
            skeletons:
            seed:
        """
        self.planning_times = []
        self.key_loc = key_loc
        self.rng = np.random.RandomState(seed)
        self.skeletons = skeletons
        self.true_h_blacklist = []
        self.goal_skel_names = goal_skel_names

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
        self.grasp_rrt.fix_start_state_in_place(phy, viz)

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
        current_locs = get_grasp_locs(phy)
        is_grasping = get_is_grasping(phy)
        for strategy in get_all_strategies_from_phy(phy):
            if not self.is_valid_strategy(strategy, is_grasping):
                continue

            # convert to numpy arrays
            for i in range(hp['n_grasp_samples']):
                # Always include the key location, 0, and 1 as candidate locations
                if i == 0:
                    sample_loc = self.key_loc
                elif i == 1:
                    sample_loc = 0
                elif i == 2 and self.key_loc != 1:
                    sample_loc = 1
                else:
                    sample_loc = self.rng.uniform(0, 1)
                candidate_locs = []
                for tool_name, s_i, loc_i in zip(phy.o.rd.tool_sites, strategy, current_locs):
                    if s_i in [Strategies.NEW_GRASP, Strategies.MOVE]:
                        candidate_locs.append(sample_loc)
                    elif s_i == Strategies.RELEASE:
                        candidate_locs.append(-1)
                    elif s_i == Strategies.STAY:
                        candidate_locs.append(loc_i)

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

            # also log as 3D boxes like a bar graph
            fig, ax = plt.subplots()
            ax.bar(['total'] + cost_names, (cost_i,) + costs_list)
            ax.set_ylim(0, BIG_PENALTY)
            strat_str = ",".join([str(s) for s in sim_grasp.strategy])
            plt.title(f'Costs for grasp {sim_grasp.locs} {strat_str}')
            plt.xticks(rotation=30)
            plt.close(fig)
            image = plt_fig_to_img_np(fig)
            rr.log_image('homotopy_costs_plot', image)

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
            return cost, 0, 0, 0, 0, 0

        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            cost = 10 * BIG_PENALTY
            return cost, 0, 0, 0, 0, 0

        homotopy_cost = 0
        if hp['use_signature_cost']:
            h_plan, loops_plan = get_full_h_signature_from_phy(self.skeletons, phy_plan)
            for blacklisted_h in self.true_h_blacklist:
                if h_plan == blacklisted_h:
                    homotopy_cost = BIG_PENALTY
                    break

        geodesics_cost = get_geodesic_dist(candidate_locs, self.key_loc) * hp['geodesic_weight']

        dq_cost = get_robot_dq_cost(res)

        rope_points0 = get_rope_points(phy0)
        rope_points_plan = get_rope_points(phy_plan)
        drope = np.linalg.norm(rope_points_plan - rope_points0, axis=-1).mean()
        drope_cost = np.clip(drope, 0, BIG_PENALTY / 2) * hp['rope_dq_weight']

        if self.goal_skel_names is None:
            goal_sig_cost = 0
        else:
            h, _ = get_full_h_signature_from_phy(self.skeletons, phy_plan)
            if h == NO_HOMOTOPY:
                goal_sig_cost = BIG_PENALTY
            else:
                h_desired = h2array(make_h_desired(self.skeletons, self.goal_skel_names))
                h = h2array(h)
                goal_sig_cost = sum(np.abs(np.array(h) - h_desired)) * BIG_PENALTY

        return 0, dq_cost, drope_cost, homotopy_cost, goal_sig_cost, geodesics_cost

    def get_cost_names(self):
        return ["penalty", "dq", "drope", "homotopy", "goal_sig_cost", "geodesic"]

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
        deactivate_release_and_moving(phy_plan, strategy, viz=viz, is_planning=True)

        # check if we need to move the arms at all
        any_moving = np.any([s in [Strategies.NEW_GRASP, Strategies.MOVE] for s in strategy])
        if any_moving:
            res, scene_msg = self.grasp_rrt.plan(phy_plan, strategy, candidate_locs, viz, max_ik_attempts=500, joint_noise=0.2)

            if res.error_code.val != MoveItErrorCodes.SUCCESS:
                return SimGraspCandidate(phy, phy_plan, strategy, res, candidate_locs, initial_locs)

            if viz_execution and viz is not None:
                self.grasp_rrt.display_result(viz, res, scene_msg)
                # print(f'dq: {get_robot_dq_cost(res):.1f}')

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

        # Activate grasps
        grasp_and_settle(phy_plan, candidate_locs, viz, is_planning=True)

        return SimGraspCandidate(phy, phy_plan, strategy, res, candidate_locs, initial_locs)

    def is_valid_strategy(self, s, is_grasping):
        will_be_grasping = [get_will_be_grasping(s_i, g_i) for s_i, g_i in zip(s, is_grasping)]
        if not any(will_be_grasping):
            return False
        if all([s_i == Strategies.STAY for s_i in s]):
            return False
        if all([s_i == Strategies.RELEASE for s_i in s]):
            return False
        if sum([s_i in [Strategies.MOVE, Strategies.NEW_GRASP] for s_i in s]) > 1:
            return False
        return True
