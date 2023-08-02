import itertools
from functools import partial
from time import perf_counter
from typing import Dict, Optional, List

import mujoco
import numpy as np
import rerun as rr
from bayes_opt import BayesianOptimization
from matplotlib import cm
from pymjregrasping_cpp import get_first_order_homotopy_points, seedOmpl

from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.grasp_conversions import grasp_locations_to_xpos
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_grasp_locs, get_is_grasping, activate_grasp
from mjregrasping.homotopy_checker import CollisionChecker, AllowablePenetration, get_full_h_signature_from_phy
from mjregrasping.ik import BIG_PENALTY
from mjregrasping.params import hp
from mjregrasping.physics import Physics, get_gripper_ctrl_indices, get_q, rescale_ctrl
from mjregrasping.rerun_visualizer import log_skeletons
from mjregrasping.rollout import DEFAULT_SUB_TIME_S, control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.viz import Viz
from moveit_msgs.msg import MoveItErrorCodes, MotionPlanResponse

GLOBAL_ITERS = 0


def _timeit(f):
    def __timeit(*args, **kwargs):
        t0 = perf_counter()
        ret = f(*args, **kwargs)
        t1 = perf_counter()
        print(f'evaluating regrasp cost: {t1 - t0:.2f}s')
        return ret
    return __timeit


def execute_grasp_plan(phy_plan: Physics, res: MotionPlanResponse, viz: Viz, is_planning):
    kp = 1.0
    prev_t = 0
    for point in res.trajectory.joint_trajectory.points:
        planned_q = np.array(point.positions)  # The position we should be at after time_from_start
        planend_qdot = np.array(point.velocities)
        current_q = get_q(phy_plan)
        next_t = point.time_from_start.to_sec()
        dt = next_t - prev_t
        pred_q = current_q + dt * planend_qdot
        print(np.rad2deg(pred_q - planned_q))
        ctrl = point.velocities + kp * (planned_q - pred_q)
        ctrl = rescale_ctrl(phy_plan, ctrl)

        control_step(phy_plan, ctrl, dt)
        if viz:
            viz.viz(phy_plan, is_planning=is_planning)
        prev_t = next_t


def release_and_settle(phy_plan, strategy, viz: Optional[Viz], is_planning: bool):
    needs_release = [s in [Strategies.MOVE, Strategies.RELEASE] for s in strategy]

    rope_grasp_eqs = phy_plan.o.rd.rope_grasp_eqs
    ctrl = np.zeros(phy_plan.m.nu)
    gripper_ctrl_indices = get_gripper_ctrl_indices(phy_plan)
    for eq_name, release_i, ctrl_i in zip(rope_grasp_eqs, needs_release, gripper_ctrl_indices):
        eq = phy_plan.m.eq(eq_name)
        if release_i:
            eq.active = 0
            ctrl[ctrl_i] = 0.5

    settle_rope(ctrl, phy_plan, viz, is_planning)


def rr_log_costs(entity_path, entity_paths, values, colors, strategy, locs):
    for path_i, v_i, color_i in zip(entity_paths, values, colors):
        rr.log_scalar(f'{entity_path}/{path_i}', v_i, color=color_i)
    rr.log_scalar(f'{entity_path}/total', sum(values), color=[1, 1, 1, 1.0])
    rr.log_tensor(f'{entity_path}_bars', values, ext={'locs': locs, 'strategy': strategy})


def get_tool_paths_from_params(phy, strategy, candidate_pos, params):
    candidate_offsets = params_to_offsets(phy, strategy, params)
    candidate_subgoals = candidate_pos[:, None] + candidate_offsets
    tool_paths = np.concatenate((candidate_pos[:, None], candidate_subgoals), axis=1)
    return tool_paths


class HomotopyRegraspPlanner:

    def __init__(self, op_goal, grasp_rrt: GraspRRT, skeletons: Dict, collision_checker: CollisionChecker, seed=0):
        self.op_goal = op_goal
        self.rng = np.random.RandomState(seed)
        self.skeletons = skeletons
        self.cc = collision_checker
        self.true_h_blacklist = []
        self.first_order_blacklist = []

        self.rrt_rng = np.random.RandomState(seed)
        self.grasp_rrt = grasp_rrt
        seedOmpl(seed)

    def generate(self, phy: Physics, viz=None):
        params, strategy = self.generate_params(phy, viz)
        locs, offsets, _ = params_to_locs_and_offsets(phy, strategy, params)
        return locs, offsets, strategy

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
            opt = BayesianOptimization(f=partial(_timeit(self.cost), strategy, phy, viz, log_loops, viz_ik),
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
             log_loops=False, viz_execution=False, **params):
        """

        Args:
            strategy: The strategy to use
            phy: Not modified
            viz: The viz object, or None if you don't want to visualize anything
            log_loops: Whether to log the loops
            viz_execution: Whether to visualize the simulation/ik
            **params: The parameters to optimize over

        Returns:
            The cost

        """
        global GLOBAL_ITERS
        rr.set_time_sequence('homotopy', GLOBAL_ITERS)
        GLOBAL_ITERS += 1

        phy_plan = phy.copy_all()

        candidate_locs, _, candidate_pos = params_to_locs_and_offsets(phy_plan, strategy, params)

        viz = viz if viz_execution else None

        # NOTE: Find a path / simulation going from the current state to the candidate_locs,
        #  then bringing those to the candidate_offsets.
        #  If we do this with the RegraspMPPI controller it will be very slow,
        #  so instead we use mujoco EQ constraints to approximate the behavior of the controller

        # If there is no significant change in the grasp, we can again skip the rest of the checks
        wrong_loc = abs(get_grasp_locs(phy_plan) - candidate_locs) > hp['grasp_loc_diff_thresh']
        should_change = strategy != Strategies.STAY
        if not np.any(wrong_loc & should_change):
            return -BIG_PENALTY

        # release and settle for any moving grippers
        release_and_settle(phy_plan, strategy, viz=viz, is_planning=True)

        res = self.grasp_rrt.plan(phy_plan, strategy, candidate_locs, viz, viz_execution)

        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            return -BIG_PENALTY

        self.grasp_rrt.display_result(res)

        plan_final_q = np.array(res.trajectory.joint_trajectory.points[-1].positions)

        # Essentially teleport the robot to the final planned joint configuration
        qpos_for_act = phy.m.actuator_trnid[:, 0]
        phy_plan.d.qpos[qpos_for_act] = plan_final_q
        phy_plan.d.act = plan_final_q
        mujoco.mj_forward(phy_plan.m, phy_plan.d)
        if viz_execution:
            viz.viz(phy_plan, is_planning=True)

        # Activate grasps
        grasp_and_settle(phy_plan, candidate_locs, viz, is_planning=True)

        # TODO: also plan the offset motions
        tool_paths = get_tool_paths_from_params(phy, strategy, candidate_pos, params)

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

        geodesics_cost = np.min(np.square(candidate_locs - self.op_goal.loc)) * hp['geodesic_weight']

        prev_plan_pos = res.trajectory.joint_trajectory.points[0].positions
        dq = 0
        for point in res.trajectory.joint_trajectory.points[1:]:
            plan_pos = point.positions
            dq += np.linalg.norm(np.array(plan_pos) - np.array(prev_plan_pos))
        dq_cost = np.clip(dq, 0, BIG_PENALTY) * hp['robot_dq_weight']

        cost = sum([
            dq_cost,
            homotopy_cost,
            geodesics_cost,
            # first_order_cost,
        ])

        # Visualize the path the tools should take for the candidate_locs and candidate_offsets
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
        print(candidate_locs, cost)

        # BayesOpt uses maximization, so we need to negate the cost
        return -cost

    def collision_free(self, tools_paths):
        # offsets has shape [n_g, T, 3]
        # and we want to figure out if the straight line path between each offset is collision free
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


def get_will_be_grasping(s: Strategies, is_grasping: bool):
    if is_grasping:
        return s in [Strategies.STAY, Strategies.MOVE]
    else:
        return s in [Strategies.NEW_GRASP]


def grasp_and_settle(phy, grasp_locs, viz: Optional[Viz], is_planning: bool):
    rope_grasp_eqs = phy.o.rd.rope_grasp_eqs
    gripper_ctrl_indices = get_gripper_ctrl_indices(phy)
    for eq_name, grasp_loc_i, ctrl_i in zip(rope_grasp_eqs, grasp_locs, gripper_ctrl_indices):
        if grasp_loc_i == -1:
            continue
        activate_grasp(phy, eq_name, grasp_loc_i)

    ctrl = np.zeros(phy.m.nu)
    settle_rope(ctrl, phy, viz, is_planning)


def settle_rope(ctrl, phy, viz: Optional[Viz], is_planning: bool):
    last_rope_points = get_rope_points(phy)
    while True:
        control_step(phy, ctrl, sub_time_s=5 * DEFAULT_SUB_TIME_S)
        rope_points = get_rope_points(phy)
        if viz:
            viz.viz(phy, is_planning=is_planning)
        rope_displacements = np.linalg.norm(rope_points - last_rope_points, axis=-1)
        if np.mean(rope_displacements) < 0.01:
            break
        last_rope_points = rope_points


def params_to_locs_and_offsets(phy: Physics, strategy, params: Dict):
    locs = params_to_locs(phy, strategy, params)
    xpos = grasp_locations_to_xpos(phy, locs)
    offsets = params_to_offsets(phy, strategy, params)
    return locs, offsets, xpos


def params_to_offsets(phy, strategy, params):
    is_grasping = get_is_grasping(phy)
    offsets = []
    for tool_name, s_i, is_grasping_i in zip(phy.o.rd.tool_sites, strategy, is_grasping):
        if s_i in [Strategies.MOVE, Strategies.NEW_GRASP]:
            offset1 = np.array([params[tool_name + '_dx_1'], params[tool_name + '_dy_1'], params[tool_name + '_dz_1']])
            offset2 = np.array([params[tool_name + '_dx_2'], params[tool_name + '_dy_2'], params[tool_name + '_dz_2']])
        else:
            offset1 = np.zeros(3)
            offset2 = np.zeros(3)
        offsets.append([offset1, offset1 + offset2])
    offsets = np.array(offsets)
    return offsets


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
    # NOTE: the below condition prevents strategies such as [NEW_GRASP, RELEASE]
    # if len(s) > 1 and any([s_i == Strategies.NEW_GRASP for s_i in s]) and any([s_i == Strategies.RELEASE for s_i in s]):
    #     return False
    return True


def duplicate_gripper_qs(phy: Physics, joint_ids, q):
    phy.o.robot.joint_names
    gripper_q_ids = get_gripper_qpos_indices(phy)
    for gripper_q_id in gripper_q_ids:
        if gripper_q_id in joint_ids:
            j = np.where(joint_ids == gripper_q_id)[0][0]
            q = np.insert(q, j + 1, q[j])
    return q

def get_gripper_qpos_indices(phy):
    gripper_q_ids = []
    for act_name in phy.o.rd.gripper_actuator_names:
        act = phy.m.actuator(act_name)
        gripper_q_ids.append(act.name)
    return gripper_q_ids
