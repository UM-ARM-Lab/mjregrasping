import itertools
from functools import partial
from time import perf_counter
from typing import Dict, Optional

import numpy as np
import rerun as rr
from bayes_opt import BayesianOptimization

from mjregrasping.eq_errors import compute_eq_errors
from mjregrasping.goal_funcs import check_should_be_open
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.grasping import get_grasp_locs, get_is_grasping, activate_grasp
from mjregrasping.homotopy_checker import get_true_homotopy_different, get_first_order_different, CollisionChecker
from mjregrasping.ik import HARD_CONSTRAINT_PENALTY, get_reachability_cost, eq_sim_ik
from mjregrasping.params import hp
from mjregrasping.physics import Physics, get_total_contact_force
from mjregrasping.rerun_visualizer import log_skeletons
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.settle import settle
from mjregrasping.viz import Viz

GLOBAL_ITERS = 0


class HomotopyRegraspPlanner:

    def __init__(self, op_goal, skeletons: Dict, collision_checker: CollisionChecker, seed=0):
        self.op_goal = op_goal
        self.rng = np.random.RandomState(seed)
        self.skeletons = skeletons
        self.cc = collision_checker

    def generate(self, phy: Physics, viz=None):
        params, is_grasping = self.generate_params(phy, viz)
        locs, subgoals, _ = params_to_locs_and_subgoals(phy, is_grasping, params)
        _, _, xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy, locs)
        return locs, subgoals, is_grasping, xpos

    def generate_params(self, phy: Physics, viz=None):
        log_skeletons(self.skeletons, stroke_width=0.01, color=[0, 1.0, 0, 1.0])

        # NOTE: what if there is no way to change homotopy? We need some kind of stopping condition.
        best_cost = np.inf
        best_params = None
        best_is_grasping = None
        for candidate_is_grasping in get_allowable_is_grasping(phy.o.rd.n_g):
            # The bounds here also define how many params there are and their names
            bounds = {}
            for tool_name, is_grasping_i in zip(phy.o.rd.tool_sites, candidate_is_grasping):
                if is_grasping_i:
                    bounds[tool_name] = (0, 1)
                    d_max = 0.15
                    bounds[tool_name + '_dx_1'] = (-d_max, d_max)
                    bounds[tool_name + '_dy_1'] = (-d_max, d_max)
                    bounds[tool_name + '_dz_1'] = (-d_max, d_max)
                    bounds[tool_name + '_dx_2'] = (-d_max, d_max)
                    bounds[tool_name + '_dy_2'] = (-d_max, d_max)
                    bounds[tool_name + '_dz_2'] = (-d_max, d_max)
            opt = BayesianOptimization(f=partial(self.cost, candidate_is_grasping, phy, viz),
                                       pbounds=bounds,
                                       verbose=0,
                                       random_state=self.rng.randint(0, 1000),
                                       allow_duplicate_points=True)
            opt.maximize(n_iter=hp['bayes_opt']['n_iter'], init_points=hp['bayes_opt']['n_init'])
            sln = opt.max
            cost = -sln['target']  # BayesOpt uses maximization, so we need to negate to get cost

            if cost < best_cost:
                best_params = sln['params']
                best_cost = cost
                best_is_grasping = candidate_is_grasping

        if best_cost > HARD_CONSTRAINT_PENALTY:
            print('No homotopy change found!')

        return best_params, best_is_grasping

    def cost(self, candidate_is_grasping: np.ndarray, phy: Physics, viz: Optional[Viz],
             log_loops=False, viz_ik=False, **params):
        """

        Args:
            candidate_is_grasping: binary array of length n_g
            phy: Not modified
            viz: The viz object, or None if you don't want to visualize anything
            log_loops: Whether to log the loops
            viz_ik: Whether to visualize the simulation/ik
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
        tool_paths = np.concatenate((candidate_pos[:, None], candidate_subgoals), axis=1)
        for tool_name, path in zip(phy_plan.o.rd.tool_sites, tool_paths):
            viz.lines(path, f'homotopy/{tool_name}_path', idx=0, scale=0.02, color=[0, 0, 1, 0.5])
            rr.log_extension_components(f'homotopy/{tool_name}_path', ext=params)

        # First check if the subgoal is straight-line collision free, because if it isn't,
        #  we waste a lot of time evaluating everything else
        if not self.collision_free(tool_paths, viz):
            return -HARD_CONSTRAINT_PENALTY

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
        def result_func(phy):
            contact_force = get_total_contact_force(phy)
            eq_err = compute_eq_errors(phy_plan)
            return contact_force, eq_err

        all_results = []
        eq_errs = []
        reached, results = eq_sim_ik(candidate_is_grasping, candidate_pos, phy_plan,
                                     viz=viz if viz_ik else None, result_func=result_func)
        all_results.extend(results)
        eq_errs.append(compute_eq_errors(phy_plan))

        if reached:
            # Activate rope grasping EQs
            for eq_name, is_grasping_i, loc_i in zip(rope_grasp_eqs, candidate_is_grasping, candidate_locs):
                if is_grasping_i:
                    activate_grasp(phy_plan, eq_name, loc_i)
            settle(phy_plan, DEFAULT_SUB_TIME_S, viz=viz if viz_ik else None, is_planning=True, result_func=result_func)
            eq_errs.append(compute_eq_errors(phy_plan))

            # 2) Do a straight line motion to the first candidate_subgoal, then the second, etc.
            # this means changing the "target" for the gripper world EQs
            for t, subgoals_t in enumerate(np.moveaxis(candidate_subgoals, 1, 0)):
                reached, results = eq_sim_ik(candidate_is_grasping, subgoals_t, phy_plan, viz=viz if viz_ik else None,
                                             result_func=result_func)
                eq_errs.append(compute_eq_errors(phy_plan))
                all_results.extend(results)
                if not reached:
                    break

        if viz_ik:
            viz.viz(phy_plan, is_planning=True)

        contact_forces, _ = zip(*all_results)

        total_accumulated_contact_force = sum(contact_forces) * hp['contact_force_weight']
        eq_err_cost = sum(eq_errs) * hp['eq_err_weight']

        reachability_cost = get_reachability_cost(phy, phy_plan, reached, candidate_locs, candidate_is_grasping)

        # This creates a sort of priority: true homotopy is more important than first order homotopy.
        homotopy_cost = 0
        if not get_true_homotopy_different(self.skeletons, phy, phy_plan, log_loops=log_loops):
            homotopy_cost += HARD_CONSTRAINT_PENALTY / 2
        if not get_first_order_different(self.cc, phy, phy_plan):
            homotopy_cost += HARD_CONSTRAINT_PENALTY / 2

        geodesics_cost = np.min(np.square(candidate_locs - self.op_goal.loc)) * hp['geodesic_weight']

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

    def collision_free(self, tools_paths, viz: Viz):
        # subgoals has shape [n_g, T, 3]
        # and we want to figure out if the straight line path between each subgoal is collision free
        lengths = np.linalg.norm(tools_paths[:, 1:] - tools_paths[:, :-1], axis=-1)
        for i, lengths_i in enumerate(lengths):
            for t in range(tools_paths.shape[1] - 1):
                for d in np.arange(0, lengths_i[t], self.cc.get_resolution() / 2):
                    p = tools_paths[i, t] + d / lengths_i[t] * (tools_paths[i, t + 1] - tools_paths[i, t])
                    in_collision = self.cc.is_collision(p)
                    if viz:
                        color = 'r' if in_collision else 'g'
                        viz.sphere(f'collision_check', p, self.cc.get_resolution() / 2, 'world', color, 0)
                    if in_collision:
                        return False
        return True


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


def get_allowable_is_grasping(n_g):
    """
    Return all possible combinations of is_grasping for n_g grippers, except for the case where no grippers are grasping
    """
    all_is_grasping = [np.array(seq) for seq in itertools.product([0, 1], repeat=n_g)]
    all_is_grasping.pop(0)
    return all_is_grasping
