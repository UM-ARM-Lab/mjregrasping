import itertools
import multiprocessing as mp
from dataclasses import dataclass
from functools import partial
from time import perf_counter
from typing import Dict, Optional, List

import numpy as np
import rerun as rr
from matplotlib import cm
from pymjregrasping_cpp import seedOmpl

from mjregrasping.goal_funcs import get_rope_points, locs_eq
from mjregrasping.goals import ObjectPointGoal
from mjregrasping.grasp_and_settle import release_and_settle, grasp_and_settle
from mjregrasping.grasp_conversions import grasp_locations_to_xpos
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_grasp_locs, get_is_grasping
from mjregrasping.homotopy_checker import get_full_h_signature_from_phy
from mjregrasping.ik import BIG_PENALTY
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import log_skeletons
from mjregrasping.rrt import GraspRRT
from mjregrasping.teleport_to_plan import teleport_to_end_of_plan
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


def rr_log_costs(entity_path, entity_paths, values, colors):
    for path_i, v_i, color_i in zip(entity_paths, values, colors):
        rr.log_scalar(f'{entity_path}/{path_i}', v_i, color=color_i)
    rr.log_scalar(f'{entity_path}/total', sum(values), color=[1, 1, 1, 1.0])


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
            sim_grasp = simulate_grasp(self.grasp_rrt, phy, viz, grasp_input, self.rng, viz_execution)
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
        same_locs = locs_eq(initial_locs - candidate_locs)
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
        rr_log_costs(entity_path='homotopy_costs', entity_paths=[
            'homotopy_costs/dq_cost',
            'homotopy_costs/homotopy_cost',
            'homotopy_costs/geodesics_cost',
        ], values=[
            dq_cost,
            homotopy_cost,
            geodesics_cost,
        ], colors=[
            [0.5, 0.5, 0, 1.0],
            [0, 1, 0, 1.0],
            [1, 0, 0, 1.0],
        ])
        # print(candidate_locs, cost)
        # rr.log_text_entry(f'homotopy_costs', f'{candidate_locs} {strategy} {cost}')

        return cost


def simulate_grasps_parallel(grasps_inputs, phy, rng):
    # NOTE: this is actually slower than the serial version, not sure why,
    #  but recreating the GraspRRT object is definitely part of the problem
    _target = partial(simulate_grasp_parallel, phy, rng)
    with mp.Pool(mp.cpu_count()) as p:
        sim_grasps = p.map(_target, grasps_inputs)
        return sim_grasps


def simulate_grasp_parallel(phy: Physics, grasp_input: SimGraspInput, rng):
    grasp_rrt = GraspRRT()
    return simulate_grasp(grasp_rrt, phy, None, grasp_input, rng, viz_execution=False)


def simulate_grasp(grasp_rrt: GraspRRT, phy: Physics, viz: Optional[Viz], grasp_input: SimGraspInput, rng,
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

    if not grasp_rrt.is_state_valid(phy_plan):
        # the RRT may fail in this case, so first try to perturb the qpos of the robot to get it out of collision
        for i in range(10):
            qpos = phy_plan.d.qpos
            # TODO: would be better to look up the jiggle_fraction ros param and use that.
            #  Although why isn't the FixStartStateAdapter fixing this problem in the first place?
            qpos[phy_plan.o.robot.qpos_indices] += np.deg2rad(
                rng.uniform(-2, 2, size=len(phy_plan.o.robot.qpos_indices)))
            valid = grasp_rrt.is_state_valid(phy_plan)
            if valid:
                break
    res, scene_msg = grasp_rrt.plan(phy_plan, strategy, candidate_locs, viz)

    if res.error_code.val != MoveItErrorCodes.SUCCESS:
        return SimGraspCandidate(phy_plan, strategy, res, candidate_locs, candidate_dxpos, tool_paths, initial_locs)

    if viz_execution:
        grasp_rrt.display_result(viz, res, scene_msg)

    # Teleport to the final planned joint configuration
    teleport_to_end_of_plan(phy_plan, res)
    if viz_execution:
        viz.viz(phy_plan, is_planning=True)

    # Activate grasps
    grasp_and_settle(phy_plan, candidate_locs, viz, is_planning=True)

    # TODO: also use RRT to plan the dxpos motions

    return SimGraspCandidate(phy_plan, strategy, res, candidate_locs, candidate_dxpos, tool_paths, initial_locs)


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
