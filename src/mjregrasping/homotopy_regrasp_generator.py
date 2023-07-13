from typing import Dict
from time import perf_counter

import numpy as np
import pysdf_tools
import rerun as rr
from bayes_opt import BayesianOptimization

from mjregrasping.eq_errors import compute_eq_errors
from mjregrasping.goal_funcs import check_should_be_open
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.grasping import get_grasp_locs, get_is_grasping, activate_grasp
from mjregrasping.homotopy_checker import HomotopyChecker
from mjregrasping.ik import HARD_CONSTRAINT_PENALTY, get_reachability_cost, eq_sim_ik
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.regrasp_generator import RegraspGenerator, get_allowable_is_grasping
from mjregrasping.rerun_visualizer import log_skeletons
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.settle import settle


class HomotopyGenerator(RegraspGenerator):

    def __init__(self, op_goal, skeletons: Dict, sdf: pysdf_tools.SignedDistanceField):
        super().__init__(op_goal)
        self.skeletons = skeletons
        self.sdf = sdf

    def generate(self, phy: Physics, viz=None):
        checker = HomotopyChecker(self.skeletons, self.sdf)
        log_skeletons(self.skeletons, stroke_width=0.01, color=[0, 1.0, 0, 1.0])

        # NOTE: what if there is no way to change homotopy? We need some kind of stopping condition.
        best_locs = None
        best_subgoals = None
        best_cost = np.inf
        for candidate_is_grasping in get_allowable_is_grasping(phy.o.rd.n_g):
            def _cost(**params):
                t0 = perf_counter()
                phy_plan = phy.copy_all()
                candidate_locs, candidate_subgoals, candidate_pos = params_to_locs_and_subgoals(phy_plan,
                                                                                                candidate_is_grasping,
                                                                                                params)

                # NOTE: Find a path / simulation going from the current state to the candidate_locs,
                # then bringing those to the candidate_subgoals.
                # If we do this with the RegraspMPPI controller it will be very slow, so we will likely need some
                # faster approximation of the dynamics (floating grippers? Virtual-Elastic Band?)

                # Visualize the path the tools should take for the candidate_locs and candidate_subgoals
                if viz:
                    tool_paths = np.concatenate((candidate_pos[:, None], candidate_subgoals), axis=1)
                    for tool_name, path in zip(phy_plan.o.rd.tool_sites, tool_paths):
                        viz.lines(path, f'homotopy/{tool_name}_path', idx=0, scale=0.02, color=[0, 0, 1, 0.5])

                # Simple approximation:
                # 1) Simulate pulling the grippers to the rope

                # First deactivate any grippers that need to change location or are simply not
                # supposed to be grasping given candidate_locs
                desired_open = check_should_be_open(current_grasp_locs=get_grasp_locs(phy_plan),
                                                    current_is_grasping=get_is_grasping(phy_plan),
                                                    desired_locs=candidate_locs,
                                                    desired_is_grasping=candidate_is_grasping)
                rope_grasp_eqs = phy_plan.o.rd.rope_grasp_eqs
                for eq_name, desired_open_i in zip(rope_grasp_eqs, desired_open):
                    eq = phy_plan.m.eq(eq_name)
                    if desired_open_i:
                        eq.active = 0

                # Use EQ constraints to pull the grippers towards the world positions corresponding to candidate_locs
                total_accumulated_contact_force = 0
                reached, accumulated_contact_force = eq_sim_ik(candidate_is_grasping, candidate_pos, phy_plan,
                                                               viz=viz)
                total_accumulated_contact_force += accumulated_contact_force

                if reached:
                    # Activate rope grasping EQs
                    for eq_name, is_grasping_i, loc_i in zip(rope_grasp_eqs, candidate_is_grasping, candidate_locs):
                        if is_grasping_i:
                            activate_grasp(phy_plan, eq_name, loc_i)
                    settle(phy_plan, DEFAULT_SUB_TIME_S, viz, is_planning=True)

                    # 2) Do a straight line motion to the first candidate_subgoal, then the second, etc.
                    # this means changing the "target" for the gripper world EQs
                    for t, subgoals_t in enumerate(np.moveaxis(candidate_subgoals, 1, 0)):
                        reached, accumulated_contact_force = eq_sim_ik(candidate_is_grasping, subgoals_t, phy_plan,
                                                                       viz=viz)
                        total_accumulated_contact_force += accumulated_contact_force
                        if not reached:
                            break

                total_accumulated_contact_force = total_accumulated_contact_force * hp['contact_force_weight']

                reachability_cost = get_reachability_cost(phy, phy_plan, reached, candidate_locs,
                                                          candidate_is_grasping)

                are_different = checker.get_true_homotopy_different(phy, phy_plan) or \
                                checker.get_first_order_different(phy, phy_plan)
                homotopy_cost = 0 if are_different else HARD_CONSTRAINT_PENALTY

                geodesics_cost = np.min(np.square(candidate_locs - self.op_goal.loc)) * hp['geodesic_weight']

                eq_err_cost = compute_eq_errors(phy_plan) * hp['eq_err_weight']

                cost = sum([
                    reachability_cost,
                    homotopy_cost,
                    geodesics_cost,
                    total_accumulated_contact_force,
                    eq_err_cost,
                ])
                rr.log_scalar('homotopy/reachability_cost', reachability_cost, color=[0, 0, 1, 1])
                rr.log_scalar('homotopy/homotopy_cost', homotopy_cost, color=[0, 1, 0, 1])
                rr.log_scalar('homotopy/geodesics_cost', geodesics_cost, color=[1, 0, 0, 1])
                rr.log_scalar('homotopy/contact_force_cost', total_accumulated_contact_force, color=[1, 0, 1, 1])
                rr.log_scalar('homotopy/eq_err_cost', eq_err_cost, color=[1, 1, 0, 1])
                rr.log_scalar('homotopy/total_cost', cost, color=[1, 1, 1, 1])

                # BayesOpt uses maximization, so we need to negate the cost
                print(f'evaluating regrasp cost: {perf_counter() - t0:.2f}s')
                return -cost

            # The bounds here also define how many params there are and their names
            bounds = {}
            for tool_name, is_grasping_i in zip(phy.o.rd.tool_sites, candidate_is_grasping):
                if is_grasping_i:
                    bounds[tool_name] = (0, 1)
                    bounds[tool_name + '_dx_1'] = (-0.25, 0.25)
                    bounds[tool_name + '_dy_1'] = (-0.25, 0.25)
                    bounds[tool_name + '_dz_1'] = (-0.25, 0.25)
                    bounds[tool_name + '_dx_2'] = (-0.25, 0.25)
                    bounds[tool_name + '_dy_2'] = (-0.25, 0.25)
                    bounds[tool_name + '_dz_2'] = (-0.25, 0.25)
            opt = BayesianOptimization(f=_cost, pbounds=bounds, verbose=0, random_state=self.rng.randint(0, 1000),
                                       allow_duplicate_points=True)
            opt.maximize(n_iter=15, init_points=5)
            sln = opt.max
            cost = -sln['target']  # BayesOpt uses maximization, so we need to negate to get cost
            locs, subgoals, _ = params_to_locs_and_subgoals(phy, candidate_is_grasping, sln['params'])

            if cost < best_cost:
                best_locs = locs
                best_subgoals = subgoals
                best_cost = cost

        return best_locs, best_subgoals


def params_to_locs_and_subgoals(phy: Physics, candidate_is_grasping, params: Dict):
    candidate_locs = []
    for tool_name, is_grasping_i in zip(phy.o.rd.tool_sites, candidate_is_grasping):
        if is_grasping_i:
            candidate_locs.append(params[tool_name])
        else:
            candidate_locs.append(-1)
    candidate_locs = np.array(candidate_locs)

    _, _, candidate_pos = grasp_locations_to_indices_and_offsets_and_xpos(phy, candidate_locs)

    candidate_subgoals = []
    for tool_name, pos_i, is_grasping_i in zip(phy.o.rd.tool_sites, candidate_pos, candidate_is_grasping):
        if is_grasping_i:
            offset1 = np.array([params[tool_name + '_dx_1'], params[tool_name + '_dy_1'], params[tool_name + '_dz_1']])
            offset2 = np.array([params[tool_name + '_dx_2'], params[tool_name + '_dy_2'], params[tool_name + '_dz_2']])
        else:
            offset1 = np.zeros(3)
            offset2 = np.zeros(3)
        candidate_subgoals.append([pos_i + offset1, pos_i + offset1 + offset2])
    candidate_subgoals = np.array(candidate_subgoals)
    return candidate_locs, candidate_subgoals, candidate_pos
