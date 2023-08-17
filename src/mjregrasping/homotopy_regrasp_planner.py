import itertools
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial
from time import perf_counter
from typing import Dict, Optional, List

import mujoco
import numpy as np
import rerun as rr
from matplotlib import cm
from pymjregrasping_cpp import get_first_order_homotopy_points, seedOmpl

from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.goals import ObjectPointGoal
from mjregrasping.grasp_conversions import grasp_locations_to_xpos
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_grasp_locs, get_is_grasping, activate_grasp
from mjregrasping.homotopy_checker import CollisionChecker, AllowablePenetration, get_full_h_signature_from_phy
from mjregrasping.ik import BIG_PENALTY
from mjregrasping.movie import MjMovieMaker
from mjregrasping.params import hp
from mjregrasping.physics import Physics, get_gripper_ctrl_indices, get_q
from mjregrasping.rerun_visualizer import log_skeletons
from mjregrasping.rollout import DEFAULT_SUB_TIME_S, control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.viz import Viz
from moveit_msgs.msg import MoveItErrorCodes, MotionPlanResponse


@dataclass
class SimGraspCandidate:
    phy: Physics
    strategy: List[Strategies]
    res: MotionPlanResponse
    locs: np.ndarray
    candidate_dxpos: np.ndarray
    tool_paths: np.ndarray
    initial_locs: np.ndarray


@dataclass
class SimGraspInput:
    strategy: List[Strategies]
    candidate_locs: np.ndarray
    candidate_dxpos: np.ndarray


def _timeit(f):
    def __timeit(*args, **kwargs):
        t0 = perf_counter()
        ret = f(*args, **kwargs)
        t1 = perf_counter()
        print(f'evaluating regrasp cost: {t1 - t0:.2f}s')
        return ret

    return __timeit


def release_and_settle(phy_plan, strategy, viz: Optional[Viz], is_planning: bool, mov: Optional[MjMovieMaker] = None):
    needs_release = [s in [Strategies.MOVE, Strategies.RELEASE] for s in strategy]

    rope_grasp_eqs = phy_plan.o.rd.rope_grasp_eqs
    ctrl = np.zeros(phy_plan.m.nu)
    gripper_ctrl_indices = get_gripper_ctrl_indices(phy_plan)
    for eq_name, release_i, ctrl_i in zip(rope_grasp_eqs, needs_release, gripper_ctrl_indices):
        eq = phy_plan.m.eq(eq_name)
        if release_i:
            eq.active = 0
            ctrl[ctrl_i] = 0.2

    settle_with_checks(ctrl, phy_plan, viz, is_planning, mov)


def rr_log_costs(entity_path, entity_paths, values, colors, strategy, locs):
    for path_i, v_i, color_i in zip(entity_paths, values, colors):
        rr.log_scalar(f'{entity_path}/{path_i}', v_i, color=color_i)
    rr.log_scalar(f'{entity_path}/total', sum(values), color=[1, 1, 1, 1.0])


class HomotopyRegraspPlanner:

    def __init__(self, op_goal: ObjectPointGoal, grasp_rrt: GraspRRT, skeletons: Dict,
                 collision_checker: CollisionChecker, seed=0):
        self.op_goal = op_goal
        self.rng = np.random.RandomState(seed)
        self.skeletons = skeletons
        self.cc = collision_checker
        self.true_h_blacklist = []
        self.first_order_blacklist = []

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

        current_path = get_rope_points(phy)
        new = True
        for blacklisted_path in self.first_order_blacklist:
            sln = get_first_order_homotopy_points(self.in_collision, blacklisted_path, current_path)
            if len(sln) > 0:
                new = False
                break
        if new:
            self.first_order_blacklist.append(current_path)

    def simulate_sampled_grasps(self, phy, viz, viz_execution=False):
        grasps_inputs = self.sample_grasp_inputs(phy)
        sim_grasps = self.simulate_grasps(grasps_inputs, phy, viz, viz_execution)
        return sim_grasps

    def simulate_grasps(self, grasps_inputs, phy, viz, viz_execution):
        sim_grasps = []
        for grasp_input in grasps_inputs:
            sim_grasp = simulate_grasp(self.grasp_rrt, phy, viz, grasp_input, viz_execution)
            sim_grasps.append(sim_grasp)
        return sim_grasps

    def sample_grasp_inputs(self, phy):
        grasps_inputs = []
        is_grasping = get_is_grasping(phy)
        for strategy in get_all_strategies_from_phy(phy):
            for i in range(20):
                if i == 0:
                    sample_loc = self.op_goal.loc
                elif i == 1:
                    sample_loc = 0
                elif i == 2:
                    sample_loc = 1
                else:
                    sample_loc = self.rng.uniform(0, 1)
                candidate_locs = []
                candidate_dxpos = []
                for tool_name, s_i, is_grasping_i in zip(phy.o.rd.tool_sites, strategy, is_grasping):
                    if s_i in [Strategies.NEW_GRASP, Strategies.MOVE]:
                        candidate_locs.append(sample_loc)
                        candidate_dxpos.append(self.rng.uniform(-0.15, 0.15, size=3))  # TODO: add waypoint
                    elif s_i in [Strategies.RELEASE, Strategies.STAY]:
                        candidate_locs.append(-1)
                        candidate_dxpos.append(np.zeros(3))

                candidate_locs = np.array(candidate_locs)
                candidate_dxpos = np.array(candidate_dxpos)

                grasps_inputs.append(SimGraspInput(strategy, candidate_locs, candidate_dxpos))
        return grasps_inputs

    def get_best(self, sim_grasps, viz: Optional[Viz], log_loops=False) -> SimGraspCandidate:
        best_cost = np.inf
        best_sim_grasp = None
        for sim_grasp in sim_grasps:
            cost_i = self.cost(sim_grasp, viz=viz, log_loops=log_loops)
            if cost_i < best_cost:
                best_cost = cost_i
                best_sim_grasp = sim_grasp

        return best_sim_grasp

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
            # print(candidate_locs, cost)
            # rr.log_text_entry(f'homotopy_costs', f'TOO CLOSE {candidate_locs} {strategy} {cost}')
            return cost

        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            cost = 10 * BIG_PENALTY
            # print(candidate_locs, cost)
            # rr.log_text_entry(f'homotopy_costs', f'NO PLAN {candidate_locs} {strategy} {cost}')
            return cost

        homotopy_cost = 0
        h_plan, loops_plan = get_full_h_signature_from_phy(self.skeletons, phy_plan)
        rope_points_plan = get_rope_points(phy_plan)
        for blacklisted_h in self.true_h_blacklist:
            if h_plan == blacklisted_h:
                homotopy_cost = BIG_PENALTY
                break

        first_order_cost = 0
        for blacklisted_rope_points in self.first_order_blacklist:
            first_order_sln = get_first_order_homotopy_points(self.in_collision, blacklisted_rope_points,
                                                              rope_points_plan)
            if len(first_order_sln) > 0:
                first_order_cost += hp['first_order_weight']  # FIXME: separate this cost term for easier debugging

        geodesics_cost = get_geodesic_dist(candidate_locs, self.op_goal) * hp['geodesic_weight']

        prev_plan_pos = res.trajectory.joint_trajectory.points[0].positions
        dq = 0
        for point in res.trajectory.joint_trajectory.points[1:]:
            plan_pos = point.positions
            dq += np.linalg.norm(np.array(plan_pos) - np.array(prev_plan_pos))
        dq_cost = np.clip(dq, 0, BIG_PENALTY / 2) * hp['robot_dq_weight']

        cost = sum([
            dq_cost,
            homotopy_cost,
            geodesics_cost,
            # first_order_cost,
        ])

        # Visualize the path the tools should take for the candidate_locs and candidate_dxpos
        if log_loops:
            rr.log_cleared(f'loops_plan', recursive=True)
        for i, l in enumerate(loops_plan):
            rr.log_line_strip(f'loops_plan/{i}', l, stroke_width=0.02)
            rr.log_extension_components(f'loops_plan/{i}', ext={'candidate_locs': f'{candidate_locs}'})
        if viz:
            viz.viz(sim_grasp.phy, is_planning=True)
            log_skeletons(self.skeletons, stroke_width=0.01, color=[0, 1.0, 0, 1.0])
            rr.log_cleared(f'homotopy', recursive=True)
            for tool_name, path in zip(phy_plan.o.rd.tool_sites, sim_grasp.tool_paths):
                color = cm.Greens(1 - min(cost, 1000) / 1000)
                viz.lines(path, f'homotopy/{tool_name}_path', idx=0, scale=0.02, color=color)
                rr.log_extension_components(f'homotopy/{tool_name}_path', ext={'locs': candidate_locs})
        rr_log_costs(
            entity_path='homotopy_costs',
            entity_paths=[
                'homotopy_costs/dq_cost',
                'homotopy_costs/homotopy_cost',
                'homotopy_costs/geodesics_cost',
            ],
            values=[
                dq_cost,
                homotopy_cost,
                geodesics_cost,
            ],
            colors=[
                [0.5, 0.5, 0, 1.0],
                [0, 1, 0, 1.0],
                [1, 0, 0, 1.0],
            ], strategy=strategy, locs=candidate_locs)
        # print(candidate_locs, cost)
        # rr.log_text_entry(f'homotopy_costs', f'{candidate_locs} {strategy} {cost}')

        return cost

    def in_collision(self, p):
        return self.cc.is_collision(p, allowable_penetration=AllowablePenetration.FULL_CELL)


def simulate_grasps_parallel(grasps_inputs, phy):
    # NOTE: this is actually slower than the serial version, not sure why,
    #  but recreating the GraspRRT object is definitely part of the problem
    _target = partial(simulate_grasp_parallel, phy)
    with mp.Pool(mp.cpu_count()) as p:
        sim_grasps = p.map(_target, grasps_inputs)
        return sim_grasps


def simulate_grasp_parallel(phy: Physics, grasp_input: SimGraspInput):
    grasp_rrt = GraspRRT()
    return simulate_grasp(grasp_rrt, phy, None, grasp_input, viz_execution=False)


def simulate_grasp(grasp_rrt: GraspRRT, phy: Physics, viz: Optional[Viz], grasp_input: SimGraspInput,
                   viz_execution=False):
    strategy = grasp_input.strategy
    candidate_locs = grasp_input.candidate_locs
    candidate_dxpos = grasp_input.candidate_dxpos
    initial_locs = get_grasp_locs(phy)

    viz = viz if viz_execution else None
    phy_plan = phy.copy_all()

    candidate_pos = grasp_locations_to_xpos(phy_plan, candidate_locs)
    tool_paths = get_tool_paths(candidate_pos, candidate_dxpos)

    # release and settle for any moving grippers
    release_and_settle(phy_plan, strategy, viz=viz, is_planning=True)

    res = grasp_rrt.plan(phy_plan, strategy, candidate_locs, viz, viz_execution)

    if res.error_code.val != MoveItErrorCodes.SUCCESS:
        return SimGraspCandidate(phy_plan, strategy, res, candidate_locs, candidate_dxpos, tool_paths, initial_locs)

    if viz_execution:
        grasp_rrt.display_result(res)

    plan_final_q = np.array(res.trajectory.joint_trajectory.points[-1].positions)

    # Teleport to the final planned joint configuration
    qpos_for_act = phy_plan.m.actuator_trnid[:, 0]
    phy_plan.d.qpos[qpos_for_act] = plan_final_q
    phy_plan.d.act = plan_final_q
    mujoco.mj_forward(phy_plan.m, phy_plan.d)
    if viz_execution:
        viz.viz(phy_plan, is_planning=True)

    # Activate grasps
    grasp_and_settle(phy_plan, candidate_locs, viz, is_planning=True)

    # TODO: also use RRT to plan the dxpos motions

    return SimGraspCandidate(phy_plan, strategy, res, candidate_locs, candidate_dxpos, tool_paths, initial_locs)


def line_collision_free(cc: CollisionChecker, tools_paths):
    # dxpos has shape [n_g, T, 3]
    # and we want to figure out if the straight line path between each dxpos is collision free
    lengths = np.linalg.norm(tools_paths[:, 1:] - tools_paths[:, :-1], axis=-1)
    for i, lengths_i in enumerate(lengths):
        for t in range(tools_paths.shape[1] - 1):
            for d in np.arange(0, lengths_i[t], cc.get_resolution() / 2):
                p = tools_paths[i, t] + d / lengths_i[t] * (tools_paths[i, t + 1] - tools_paths[i, t])
                in_collision = cc.is_collision(p, allowable_penetration=AllowablePenetration.HALF_CELL)
                if in_collision:
                    return False
    return True


def get_tool_paths(candidate_pos, candidate_dxpos):
    candidate_subgoals = candidate_pos[:, None] + candidate_dxpos
    tool_paths = np.concatenate((candidate_pos[:, None], candidate_subgoals), axis=1)
    return tool_paths


def get_geodesic_dist(locs, op_goal: ObjectPointGoal):
    return np.min(np.abs(locs - op_goal.loc))


def get_will_be_grasping(s: Strategies, is_grasping: bool):
    if is_grasping:
        return s in [Strategies.STAY, Strategies.MOVE]
    else:
        return s in [Strategies.NEW_GRASP]


def grasp_and_settle(phy, grasp_locs, viz: Optional[Viz], is_planning: bool, mov: Optional[MjMovieMaker] = None):
    rope_grasp_eqs = phy.o.rd.rope_grasp_eqs
    ctrl = np.zeros(phy.m.nu)
    gripper_ctrl_indices = get_gripper_ctrl_indices(phy)
    for eq_name, grasp_loc_i, ctrl_i in zip(rope_grasp_eqs, grasp_locs, gripper_ctrl_indices):
        if grasp_loc_i == -1:
            continue
        ctrl[ctrl_i] = -0.5
        activate_grasp(phy, eq_name, grasp_loc_i)

    settle_with_checks(ctrl, phy, viz, is_planning, mov)


def settle_with_checks(ctrl, phy, viz: Optional[Viz], is_planning: bool, mov: Optional[MjMovieMaker] = None):
    """
    In contrast to settle(), which steps for a fixed number of steps, this function steps until the rope and robot
    have settled.
    """
    last_rope_points = get_rope_points(phy)
    last_q = get_q(phy)
    max_t = 40
    for t in range(max_t):
        control_step(phy, ctrl, sub_time_s=5 * DEFAULT_SUB_TIME_S)
        rope_points = get_rope_points(phy)
        q = get_q(phy)
        if viz:
            viz.viz(phy, is_planning=is_planning)
        if mov:
            mov.render(phy.d)
        rope_displacements = np.linalg.norm(rope_points - last_rope_points, axis=-1)
        robot_displacements = np.linalg.norm(q - last_q)
        is_unstable = phy.d.warning.number.sum() > 0
        if np.mean(rope_displacements) < 0.01 and np.mean(robot_displacements) < np.deg2rad(1) or is_unstable:
            return
        last_rope_points = rope_points
        last_q = q
    if not is_planning:
        print(f'WARNING: settle_with_checks failed to settle after {max_t} steps')


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
    # NOTE: the below condition prevents strategies such as [NEW_GRASP, RELEASE]
    # if len(s) > 1 and any([s_i == Strategies.NEW_GRASP for s_i in s]) and any([s_i == Strategies.RELEASE for s_i in s]):
    #     return False
    return True
