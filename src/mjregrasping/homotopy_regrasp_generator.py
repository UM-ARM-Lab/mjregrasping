from functools import partial
from typing import Dict, Optional
from time import perf_counter, time

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
from mjregrasping.viz import Viz

GLOBAL_ITERS = 0


class HomotopyGenerator(RegraspGenerator):

    def __init__(self, op_goal, skeletons: Dict, sdf: pysdf_tools.SignedDistanceField, seed=0):
        super().__init__(op_goal, seed)
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
            opt = BayesianOptimization(f=partial(self.cost, checker, candidate_is_grasping, phy, viz),
                                       pbounds=bounds,
                                       verbose=0,
                                       random_state=self.rng.randint(0, 1000),
                                       allow_duplicate_points=True)
            opt.maximize(n_iter=15, init_points=5)
            sln = opt.max
            cost = -sln['target']  # BayesOpt uses maximization, so we need to negate to get cost
            locs, subgoals, _ = params_to_locs_and_subgoals(phy, candidate_is_grasping, sln['params'])

            if cost < best_cost:
                best_locs = locs
                best_subgoals = subgoals
                best_cost = cost

        if best_cost > HARD_CONSTRAINT_PENALTY:
            print('No homotopy change found!')

        return best_locs, best_subgoals

    def cost(self, checker: HomotopyChecker, candidate_is_grasping: np.ndarray, phy: Physics, viz: Optional[Viz],
             **params):
        """

        Args:
            checker: The homotopy checker
            candidate_is_grasping: binary array of length n_g
            phy: Not modified
            viz: The viz object, or None if you don't want to visualize anything
            **params: The parameters to optimize over

        Returns:
            The cost

        """
        global GLOBAL_ITERS
        rr.set_time_sequence('homotopy', GLOBAL_ITERS)
        GLOBAL_ITERS += 1
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
        desired_open = check_should_be_open(current_grasp_locs=get_grasp_locs(phy),
                                            current_is_grasping=get_is_grasping(phy),
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
                                                       viz=None)
        total_accumulated_contact_force += accumulated_contact_force

        if reached:
            # Activate rope grasping EQs
            for eq_name, is_grasping_i, loc_i in zip(rope_grasp_eqs, candidate_is_grasping, candidate_locs):
                if is_grasping_i:
                    activate_grasp(phy_plan, eq_name, loc_i)
            settle(phy_plan, DEFAULT_SUB_TIME_S, viz=None, is_planning=True)

            # 2) Do a straight line motion to the first candidate_subgoal, then the second, etc.
            # this means changing the "target" for the gripper world EQs
            for t, subgoals_t in enumerate(np.moveaxis(candidate_subgoals, 1, 0)):
                reached, accumulated_contact_force = eq_sim_ik(candidate_is_grasping, subgoals_t, phy_plan,
                                                               viz=None)
                total_accumulated_contact_force += accumulated_contact_force
                if not reached:
                    break

        viz.viz(phy_plan, is_planning=True)

        total_accumulated_contact_force = total_accumulated_contact_force * hp['contact_force_weight']

        reachability_cost = get_reachability_cost(phy, phy_plan, reached, candidate_locs,
                                                  candidate_is_grasping)

        # If it's different by true homotopy, it is also different by first order homotopy
        # so the homotopy_cost will be 0. If it's not different by true homotopy, it may still
        # be different by first order homotopy, in which case the homotopy_cost will be HARD_CONSTRAINT_PENALTY.
        # If it's not different by first order homotopy or true homotopy, the cost is 2 * HARD_CONSTRAINT_PENALTY.
        # This creates a sort of priority: true homotopy is more important than first order homotopy.
        homotopy_cost = 0
        if not checker.get_true_homotopy_different(phy, phy_plan, log_loops=(viz is not None)):
            homotopy_cost += HARD_CONSTRAINT_PENALTY
        if not checker.get_first_order_different(phy, phy_plan):
            homotopy_cost += HARD_CONSTRAINT_PENALTY

        geodesics_cost = np.min(np.square(candidate_locs - self.op_goal.loc)) * hp['geodesic_weight']

        eq_err_cost = compute_eq_errors(phy_plan) * hp['eq_err_weight']

        unstable_cost = (phy.d.warning.number.sum() > 0) * HARD_CONSTRAINT_PENALTY

        cost = sum([
            reachability_cost,
            homotopy_cost,
            geodesics_cost,
            total_accumulated_contact_force,
            eq_err_cost,
            unstable_cost,
        ])
        rr.log_scalar('homotopy_costs/reachability_cost', reachability_cost, color=[0, 0, 1, 1.0])
        rr.log_scalar('homotopy_costs/homotopy_cost', homotopy_cost, color=[0, 1, 0, 1.0])
        rr.log_scalar('homotopy_costs/geodesics_cost', geodesics_cost, color=[1, 0, 0, 1.0])
        rr.log_scalar('homotopy_costs/contact_force_cost', total_accumulated_contact_force, color=[1, 0, 1, 1.0])
        rr.log_scalar('homotopy_costs/eq_err_cost', eq_err_cost, color=[1, 1, 0, 1.0])
        rr.log_scalar('homotopy_costs/unstable_cost', unstable_cost, color=[1, 0.5, 0, 1.0])
        rr.log_scalar('homotopy_costs/total_cost', cost, color=[1, 1, 1, 1.0])

        t1 = perf_counter()
        # print(f'evaluating regrasp cost: {t1 - t0:.2f}s')

        # BayesOpt uses maximization, so we need to negate the cost
        return -cost


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
