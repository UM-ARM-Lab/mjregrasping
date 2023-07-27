import itertools
from enum import auto, Enum
from functools import partial
from time import perf_counter
from typing import Dict, Optional, List

import numpy as np
import rerun as rr
from bayes_opt import BayesianOptimization
from matplotlib import cm
from pymjregrasping_cpp import get_first_order_homotopy_points, RRTPlanner

from mjregrasping.eq_errors import compute_eq_errors
from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.grasping import get_grasp_locs, get_is_grasping, activate_grasp
from mjregrasping.homotopy_checker import CollisionChecker, AllowablePenetration, get_full_h_signature_from_phy
from mjregrasping.ik import BIG_PENALTY, get_reachability_cost, eq_sim_ik
from mjregrasping.params import hp
from mjregrasping.physics import Physics, get_total_contact_force
from mjregrasping.rerun_visualizer import log_skeletons
from mjregrasping.rollout import DEFAULT_SUB_TIME_S, control_step
from mjregrasping.settle import settle
from mjregrasping.viz import Viz

GLOBAL_ITERS = 0


class Strategies(Enum):
    NEW_GRASP = auto()
    RELEASE = auto()
    MOVE = auto()
    STAY = auto()


def get_will_be_grasping(s: Strategies, is_grasping: bool):
    if is_grasping:
        return s in [Strategies.STAY, Strategies.MOVE]
    else:
        return s in [Strategies.NEW_GRASP]


def release_and_settle(phy_plan, release, viz, viz_ik, result_func):
    results = []
    rope_grasp_eqs = phy_plan.o.rd.rope_grasp_eqs
    ctrl = np.zeros(phy_plan.m.nu)
    gripper_ctrl_indices = [phy_plan.m.actuator(actuator_name).id for actuator_name in
                            phy_plan.o.rd.gripper_actuator_names]
    for eq_name, release_i, ctrl_i in zip(rope_grasp_eqs, release, gripper_ctrl_indices):
        eq = phy_plan.m.eq(eq_name)
        if release_i:
            eq.active = 0
            ctrl[ctrl_i] = 0.5

    if any(release):
        last_rope_points = get_rope_points(phy_plan)
        while True:
            control_step(phy_plan, ctrl, sub_time_s=5 * DEFAULT_SUB_TIME_S)
            rope_points = get_rope_points(phy_plan)
            results.append(result_func(phy_plan))
            if viz_ik:
                viz.viz(phy_plan, is_planning=True)
            rope_displacements = np.linalg.norm(rope_points - last_rope_points, axis=-1)
            if np.mean(rope_displacements) < 0.01:
                break
            last_rope_points = rope_points

    return results


class HomotopyRegraspPlanner:

    def __init__(self, op_goal, skeletons: Dict, collision_checker: CollisionChecker, seed=0):
        self.op_goal = op_goal
        self.rng = np.random.RandomState(seed)
        self.skeletons = skeletons
        self.cc = collision_checker
        self.true_h_blacklist = []
        self.first_order_blacklist = []

        self.rrt_planner = RRTPlanner()

    def generate(self, phy: Physics, viz=None):
        params, strategy = self.generate_params(phy, viz)
        locs, subgoals, _ = params_to_locs_and_subgoals(phy, strategy, params)
        _, _, xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy, locs)

        return locs, subgoals

    def generate_params(self, phy: Physics, viz=None, log_loops=False, viz_ik=False):
        self.update_blacklists(phy)

        log_skeletons(self.skeletons, stroke_width=0.01, color=[0, 1.0, 0, 1.0])

        # Parallelize this
        best_cost = np.inf
        best_params = None
        best_strategy = None
        is_grasping = get_is_grasping(phy)
        for strategy in get_all_strategies_from_phy(phy):
            # The bounds here also define how many params there are and their names
            bounds = {}
            for tool_name, s_i, is_grasping_i in zip(phy.o.rd.tool_sites, strategy, is_grasping):
                if s_i in [Strategies.NEW_GRASP, Strategies.MOVE]:
                    bounds[tool_name] = (0, 1)
                    d_max = 0.15
                    bounds[tool_name + '_dx_1'] = (-d_max, d_max)
                    bounds[tool_name + '_dy_1'] = (-d_max, d_max)
                    bounds[tool_name + '_dz_1'] = (-d_max, d_max)
                    bounds[tool_name + '_dx_2'] = (-d_max, d_max)
                    bounds[tool_name + '_dy_2'] = (-d_max, d_max)
                    bounds[tool_name + '_dz_2'] = (-d_max, d_max)
                elif s_i == Strategies.RELEASE:
                    pass  # no decision to make here
                elif s_i == Strategies.STAY:
                    pass  # Currently testing what happens if we do not plan a delta motion?
            opt = BayesianOptimization(f=partial(self.cost, strategy, phy, viz, log_loops, viz_ik),
                                       pbounds=bounds,
                                       verbose=0,
                                       random_state=self.rng.randint(0, 1000),
                                       allow_duplicate_points=True)
            opt.maximize(n_iter=hp['bayes_opt']['n_iter'], init_points=hp['bayes_opt']['n_init'])
            sln = opt.max
            cost = -sln['target']  # BayesOpt uses maximization, so we need to negate to get cost

            locs = params_to_locs(phy, strategy, sln['params'])
            print(f'{strategy} cost={cost} locs={locs}')

            if cost < best_cost:
                best_params = sln['params']
                best_cost = cost
                best_strategy = strategy

        if best_cost > BIG_PENALTY:
            print('No homotopy change found!')

        return best_params, best_strategy

    def update_blacklists(self, phy):
        current_true_h, _ = get_full_h_signature_from_phy(self.skeletons, phy)
        new = True
        for blacklisted_true_h in self.true_h_blacklist:
            if current_true_h == blacklisted_true_h:
                new = False
                break
        if new:
            self.true_h_blacklist.append(current_true_h)

        current_path = get_rope_points(phy)
        new = True
        for blacklisted_path in self.first_order_blacklist:
            sln = get_first_order_homotopy_points(self.in_collision, blacklisted_path, current_path)
            if len(sln) > 0:
                new = False
                break
        if new:
            self.first_order_blacklist.append(current_path)

    def cost(self, strategy: List[Strategies], phy: Physics, viz: Optional[Viz],
             log_loops=False, viz_ik=False, **params):
        """

        Args:
            strategy: The strategy to use
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

        candidate_locs, candidate_subgoals, candidate_pos = params_to_locs_and_subgoals(phy_plan, strategy, params)
        candidate_is_grasping = (candidate_locs != -1)

        # NOTE: Find a path / simulation going from the current state to the candidate_locs,
        #  then bringing those to the candidate_subgoals.
        #  If we do this with the RegraspMPPI controller it will be very slow,
        #  so instead we use mujoco EQ constraints to approximate the behavior of the controller

        tool_paths = np.concatenate((candidate_pos[:, None], candidate_subgoals), axis=1)

        # First check if the subgoal is straight-line collision free, because if it isn't,
        #  we waste a lot of time evaluating everything else
        if not self.collision_free(tool_paths):
            return -10 * BIG_PENALTY

        # If there is no significant change in the grasp, we can again skip the rest of the checks
        wrong_loc = abs(get_grasp_locs(phy) - candidate_locs) > hp['grasp_loc_diff_thresh']
        should_change = strategy != Strategies.STAY
        if not np.any(wrong_loc & should_change):
            return -10 * BIG_PENALTY

        if viz:
            rr.log_cleared(f'homotopy', recursive=True)
            for tool_name, path in zip(phy_plan.o.rd.tool_sites, tool_paths):
                viz.lines(path, f'homotopy/{tool_name}_path', idx=0, scale=0.02, color=[0, 0, 1, 0.5])
                rr.log_extension_components(f'homotopy/{tool_name}_path', ext=params)

        _, _, candidate_pos = grasp_locations_to_indices_and_offsets_and_xpos(phy_plan, candidate_locs)

        # Run RRT to find a collision free path from the current q to candidate_pos
        # Then in theory we should do it for the subgoals to
        q0 = phy_plan.d.qpos[phy_plan.o.robot.qpos_indices]
        strategy_strs = [s.name for s in strategy]
        is_grasping0 = get_is_grasping(phy_plan)
        reached, trajectory = plan_from_q_to_target_pos(phy_plan, q0, candidate_pos, strategy_strs, is_grasping0)
        q_final = np.array(trajectory.points[-1].position)

        reachability_cost = BIG_PENALTY if reached else 0

        homotopy_cost = 0
        h_plan, loops_plan = get_full_h_signature_from_phy(self.skeletons, phy_plan)
        rope_points_plan = get_rope_points(phy_plan)
        for blacklisted_h in self.true_h_blacklist:
            if h_plan == blacklisted_h:
                homotopy_cost += BIG_PENALTY / len(self.true_h_blacklist)

        for blacklisted_rope_points in self.first_order_blacklist:
            first_order_sln = get_first_order_homotopy_points(self.in_collision, blacklisted_rope_points,
                                                              rope_points_plan)
            if len(first_order_sln) > 0:
                homotopy_cost += hp['first_order_weight']

        geodesics_cost = np.min(np.square(candidate_locs - self.op_goal.loc)) * hp['geodesic_weight']
        dq_cost = np.sum(np.linalg.norm(q0 - q_final, axis=-1)) * hp['robot_dq_weight']

        cost = sum([
            dq_cost,
            reachability_cost,
            homotopy_cost,
            geodesics_cost,
        ])

        # Visualize the path the tools should take for the candidate_locs and candidate_subgoals
        if log_loops:
            rr.log_cleared(f'loops_plan', recursive=True)
        for i, l in enumerate(loops_plan):
            rr.log_line_strip(f'loops_plan/{i}', l, stroke_width=0.02)
            rr.log_extension_components(f'loops_plan/{i}', ext={'candidate_locs': f'{candidate_locs}'})
        if viz:
            rr.log_cleared(f'homotopy', recursive=True)
            for tool_name, path in zip(phy_plan.o.rd.tool_sites, tool_paths):
                color = cm.Greens(1 - min(cost, 1000) / 1000)
                viz.lines(path, f'homotopy/{tool_name}_path', idx=0, scale=0.02, color=color)
                rr.log_extension_components(f'homotopy/{tool_name}_path', ext=params)
        rr.log_scalar('homotopy_costs/dq_cost', dq_cost, color=[0.5, 0.5, 0, 1.0])
        rr.log_scalar('homotopy_costs/reachability_cost', reachability_cost, color=[0, 0, 1, 1.0])
        rr.log_scalar('homotopy_costs/homotopy_cost', homotopy_cost, color=[0, 1, 0, 1.0])
        rr.log_scalar('homotopy_costs/geodesics_cost', geodesics_cost, color=[1, 0, 0, 1.0])
        rr.log_scalar('homotopy_costs/total_cost', cost, color=[1, 1, 1, 1.0])
        print(candidate_locs, cost)

        t1 = perf_counter()
        # print(f'evaluating regrasp cost: {t1 - t0:.2f}s')

        # BayesOpt uses maximization, so we need to negate the cost
        return -cost

    def cost_mpc(self, strategy: List[Strategies], phy: Physics, viz: Optional[Viz],
             log_loops=False, viz_ik=False, **params):
        """

        Args:
            strategy: The strategy to use
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

        candidate_locs, candidate_subgoals, candidate_pos = params_to_locs_and_subgoals(phy_plan, strategy, params)
        candidate_is_grasping = (candidate_locs != -1)

        # NOTE: Find a path / simulation going from the current state to the candidate_locs,
        #  then bringing those to the candidate_subgoals.
        #  If we do this with the RegraspMPPI controller it will be very slow,
        #  so instead we use mujoco EQ constraints to approximate the behavior of the controller

        tool_paths = np.concatenate((candidate_pos[:, None], candidate_subgoals), axis=1)

        # First check if the subgoal is straight-line collision free, because if it isn't,
        #  we waste a lot of time evaluating everything else
        if not self.collision_free(tool_paths):
            return -10 * BIG_PENALTY

        # If there is no significant change in the grasp, we can again skip the rest of the checks
        wrong_loc = abs(get_grasp_locs(phy) - candidate_locs) > hp['grasp_loc_diff_thresh']
        should_change = strategy != Strategies.STAY
        if not np.any(wrong_loc & should_change):
            return -10 * BIG_PENALTY

        if viz:
            rr.log_cleared(f'homotopy', recursive=True)
            for tool_name, path in zip(phy_plan.o.rd.tool_sites, tool_paths):
                viz.lines(path, f'homotopy/{tool_name}_path', idx=0, scale=0.02, color=[0, 0, 1, 0.5])
                rr.log_extension_components(f'homotopy/{tool_name}_path', ext=params)

        def result_func(_phy):
            contact_force = get_total_contact_force(_phy)
            q = _phy.d.qpos[_phy.o.robot.qpos_indices]
            rope_points = get_rope_points(_phy)
            return contact_force, q, rope_points

        all_results = []

        # First release any grippers that need to move
        results = release_and_settle(phy_plan, wrong_loc, viz, viz_ik, result_func)
        all_results.extend(results)

        eq_errs = []

        _, _, candidate_pos = grasp_locations_to_indices_and_offsets_and_xpos(phy_plan, candidate_locs)

        # Simulate the grasp
        reached, results = eq_sim_ik(candidate_is_grasping, candidate_pos, phy_plan,
                                     viz=viz if viz_ik else None, result_func=result_func)
        all_results.extend(results)
        eq_errs.append(compute_eq_errors(phy_plan))

        if reached:
            # Activate rope grasping EQs
            rope_grasp_eqs = phy_plan.o.rd.rope_grasp_eqs
            for eq_name, is_grasping_i, loc_i in zip(rope_grasp_eqs, candidate_is_grasping, candidate_locs):
                if is_grasping_i:
                    activate_grasp(phy_plan, eq_name, loc_i)
            # Settle
            settle(phy_plan, DEFAULT_SUB_TIME_S, viz=viz if viz_ik else None, is_planning=True, result_func=result_func)
            eq_errs.append(compute_eq_errors(phy_plan))

            # Now release any other grippers that need to release
            needs_release = strategy == Strategies.RELEASE
            results = release_and_settle(phy_plan, needs_release, viz, viz_ik, result_func)
            all_results.extend(results)

            # Do a straight line motions to the subgoals
            for t, subgoals_t in enumerate(np.moveaxis(candidate_subgoals, 1, 0)):
                reached, results = eq_sim_ik(candidate_is_grasping, subgoals_t, phy_plan, viz=viz if viz_ik else None,
                                             result_func=result_func)
                eq_errs.append(compute_eq_errors(phy_plan))
                all_results.extend(results)
                if not reached:
                    break

        if viz_ik:
            viz.viz(phy_plan, is_planning=True)

        contact_forces, qs, rope_points = zip(*all_results)
        contact_forces = np.array(contact_forces)
        qs = np.array(qs)
        rope_points = np.array(rope_points)

        total_accumulated_contact_force = min(sum(contact_forces) * hp['contact_force_weight'], BIG_PENALTY)
        eq_err_cost = sum(eq_errs) * hp['eq_err_weight']

        reachability_cost = get_reachability_cost(phy, phy_plan, reached, candidate_locs, candidate_is_grasping)

        dq_cost = np.sum(np.linalg.norm(qs[1:] - qs[:-1], axis=-1)) * hp['robot_dq_weight']
        rope_displacements = np.linalg.norm(rope_points[1:] - rope_points[:-1], axis=-1)
        rope_disp_cost = np.sum(np.mean(rope_displacements, axis=-1)) * hp['rope_disp_weight']

        homotopy_cost = 0
        h_plan, loops_plan = get_full_h_signature_from_phy(self.skeletons, phy_plan)
        rope_points_plan = get_rope_points(phy_plan)
        for blacklisted_h in self.true_h_blacklist:
            if h_plan == blacklisted_h:
                homotopy_cost += BIG_PENALTY / len(self.true_h_blacklist)

        for blacklisted_rope_points in self.first_order_blacklist:
            first_order_sln = get_first_order_homotopy_points(self.in_collision, blacklisted_rope_points,
                                                              rope_points_plan)
            if len(first_order_sln) > 0:
                homotopy_cost += hp['first_order_weight']

        geodesics_cost = np.min(np.square(candidate_locs - self.op_goal.loc)) * hp['geodesic_weight']

        unstable_cost = (phy.d.warning.number.sum() > 0) * BIG_PENALTY

        cost = sum([
            dq_cost,  # sort of a reachability cost
            reachability_cost,
            homotopy_cost,
            geodesics_cost,
            total_accumulated_contact_force,
            eq_err_cost,
            unstable_cost,
            rope_disp_cost,
        ])

        # Visualize the path the tools should take for the candidate_locs and candidate_subgoals
        if log_loops:
            rr.log_cleared(f'loops_plan', recursive=True)
        for i, l in enumerate(loops_plan):
            rr.log_line_strip(f'loops_plan/{i}', l, stroke_width=0.02)
            rr.log_extension_components(f'loops_plan/{i}', ext={'candidate_locs': f'{candidate_locs}'})
        if viz:
            rr.log_cleared(f'homotopy', recursive=True)
            for tool_name, path in zip(phy_plan.o.rd.tool_sites, tool_paths):
                color = cm.Greens(1 - min(cost, 1000) / 1000)
                viz.lines(path, f'homotopy/{tool_name}_path', idx=0, scale=0.02, color=color)
                rr.log_extension_components(f'homotopy/{tool_name}_path', ext=params)
        rr.log_scalar('homotopy_costs/dq_cost', dq_cost, color=[0.5, 0.5, 0, 1.0])
        rr.log_scalar('homotopy_costs/reachability_cost', reachability_cost, color=[0, 0, 1, 1.0])
        rr.log_scalar('homotopy_costs/homotopy_cost', homotopy_cost, color=[0, 1, 0, 1.0])
        rr.log_scalar('homotopy_costs/geodesics_cost', geodesics_cost, color=[1, 0, 0, 1.0])
        rr.log_scalar('homotopy_costs/contact_force_cost', total_accumulated_contact_force, color=[1, 0, 1, 1.0])
        rr.log_scalar('homotopy_costs/eq_err_cost', eq_err_cost, color=[1, 1, 0, 1.0])
        rr.log_scalar('homotopy_costs/unstable_cost', unstable_cost, color=[1, 0.5, 0, 1.0])
        rr.log_scalar('homotopy_costs/rope_dq_cost', rope_disp_cost, color=[0, 1, 1, 1.0])
        rr.log_scalar('homotopy_costs/total_cost', cost, color=[1, 1, 1, 1.0])
        print(candidate_locs, cost)

        t1 = perf_counter()
        # print(f'evaluating regrasp cost: {t1 - t0:.2f}s')

        # BayesOpt uses maximization, so we need to negate the cost
        return -cost

    def collision_free(self, tools_paths):
        # subgoals has shape [n_g, T, 3]
        # and we want to figure out if the straight line path between each subgoal is collision free
        lengths = np.linalg.norm(tools_paths[:, 1:] - tools_paths[:, :-1], axis=-1)
        for i, lengths_i in enumerate(lengths):
            for t in range(tools_paths.shape[1] - 1):
                for d in np.arange(0, lengths_i[t], self.cc.get_resolution() / 2):
                    p = tools_paths[i, t] + d / lengths_i[t] * (tools_paths[i, t + 1] - tools_paths[i, t])
                    in_collision = self.cc.is_collision(p, allowable_penetration=AllowablePenetration.HALF_CELL)
                    if in_collision:
                        return False
        return True

    def in_collision(self, p):
        return self.cc.is_collision(p, allowable_penetration=AllowablePenetration.FULL_CELL)


def params_to_locs_and_subgoals(phy: Physics, strategy, params: Dict):
    is_grasping = get_is_grasping(phy)
    candidate_locs = params_to_locs(phy, strategy, params)

    _, _, candidate_pos = grasp_locations_to_indices_and_offsets_and_xpos(phy, candidate_locs)

    candidate_subgoals = []
    for tool_name, pos_i, s_i, is_grasping_i in zip(phy.o.rd.tool_sites, candidate_pos, strategy, is_grasping):
        if s_i in [Strategies.MOVE, Strategies.NEW_GRASP]:
            offset1 = np.array([params[tool_name + '_dx_1'], params[tool_name + '_dy_1'], params[tool_name + '_dz_1']])
            offset2 = np.array([params[tool_name + '_dx_2'], params[tool_name + '_dy_2'], params[tool_name + '_dz_2']])
        else:
            offset1 = np.zeros(3)
            offset2 = np.zeros(3)
        candidate_subgoals.append([pos_i + offset1, pos_i + offset1 + offset2])
    candidate_subgoals = np.array(candidate_subgoals)
    return candidate_locs, candidate_subgoals, candidate_pos


def params_to_locs(phy: Physics, strategy, params: Dict):
    candidate_locs = []
    is_grasping = get_is_grasping(phy)
    locs = get_grasp_locs(phy)
    for tool_name, s_i, is_grasping_i, loc_i in zip(phy.o.rd.tool_sites, strategy, is_grasping, locs):
        if get_will_be_grasping(s_i, is_grasping_i):
            if s_i == Strategies.STAY:
                candidate_locs.append(loc_i)
            else:
                candidate_locs.append(params[tool_name])
        else:
            candidate_locs.append(-1)
    candidate_locs = np.array(candidate_locs)
    return candidate_locs


def get_all_strategies_from_phy(phy: Physics):
    current_is_grasping = get_is_grasping(phy)
    return get_all_strategies(phy.o.rd.n_g, current_is_grasping)


def get_all_strategies(n_g: int, current_is_grasping: np.ndarray):
    strategies_per_gripper = []
    for i in range(n_g):
        is_grasping_i = current_is_grasping[i]
        strategies = []
        for strategy in Strategies:
            if strategy == Strategies.NEW_GRASP:
                if is_grasping_i:
                    continue  # not valid
                else:
                    strategies.append(strategy)
            elif strategy == Strategies.RELEASE:
                if not is_grasping_i:
                    continue  # not valid
                else:
                    strategies.append(strategy)
            elif strategy == Strategies.MOVE:
                if not is_grasping_i:
                    continue  # not valid
                else:
                    strategies.append(strategy)
            elif strategy == Strategies.STAY:
                strategies.append(strategy)
            else:
                raise NotImplementedError(strategy)
        strategies_per_gripper.append(strategies)

    all_strategies = list(itertools.product(*strategies_per_gripper))

    # filter out invalid strategies
    all_strategies = [s_i for s_i in all_strategies if is_valid_strategy(s_i, current_is_grasping)]
    # convert to numpy arrays
    all_strategies = np.array(all_strategies)
    return all_strategies


def is_valid_strategy(s, is_grasping):
    will_be_grasping = [get_will_be_grasping(s_i, g_i) for s_i, g_i in zip(s, is_grasping)]
    if not any(will_be_grasping):
        return False
    if all([s_i == Strategies.STAY for s_i in s]):
        return False
    if all([s_i == Strategies.RELEASE for s_i in s]):
        return False
    if sum([s_i in [Strategies.MOVE, Strategies.NEW_GRASP] for s_i in s]) > 1:
        return False
    if len(s) > 1 and any([s_i == Strategies.NEW_GRASP for s_i in s]) and any([s_i == Strategies.RELEASE for s_i in s]):
        return False
    return True
