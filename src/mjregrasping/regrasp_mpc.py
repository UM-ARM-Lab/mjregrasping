import itertools
import logging
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from typing import Optional, Dict, List

import cma
import mujoco
import numpy as np
import rerun as rr
from colorama import Fore
from matplotlib import cm
from numpy.linalg import norm

import rospy
from mjregrasping.body_with_children import Objects
from mjregrasping.buffer import Buffer
from mjregrasping.change_grasp_eq import change_eq
from mjregrasping.goal_funcs import get_tool_positions
from mjregrasping.goals import ObjectPointGoal, GraspRopeGoal, get_contact_cost, RegraspGoal
from mjregrasping.grasp_state import GraspState
from mjregrasping.grasp_state_utils import grasp_location_to_indices, grasp_indices_to_locations, grasp_offset
from mjregrasping.grasping import deactivate_eq, compute_eq_errors, gripper_idx_to_eq_name, activate_grasp
from mjregrasping.mjsaver import save_data_and_eq
from mjregrasping.movie import MjMovieMaker
from mjregrasping.mujoco_mppi import MujocoMPPI
from mjregrasping.params import hp, Params
from mjregrasping.physics import Physics
from mjregrasping.regrasping_mppi import RegraspMPPI, do_grasps_if_close, release_dynamics
from mjregrasping.rerun_visualizer import log_box
from mjregrasping.rollout import control_step, rollout, expand_result
from mjregrasping.settle import settle
from mjregrasping.viz import Viz

GRASP_POINT_OFFSET = 0.17

logger = logging.getLogger(f'rosout.{__name__}')


class Status(Enum):
    SUCCESS = auto()
    REGRASP = auto()
    FAILED = auto()


class Result:

    def __init__(self, status: Status, msg: Optional[str] = None, cost: Optional = None):
        self.status = status
        self.msg = msg
        self.cost = cost

    def __str__(self):
        return f"{self.status.name} {self.msg}"

    def __repr__(self):
        return str(self)


def viz_regrasp_solutions_and_costs(costs_dicts: List[Dict], candidate_grasps: List[GraspState]):
    # histograms
    n_g = 2
    width = 0.2
    cost_to_meters = 0.5
    depth = 0.05
    cost_colors = {
        'f_is_same':             [1.0, 0, 1.0],
        'f_all_0':               [1.0, 0, 1.0],
        'f_goal':                [0.5, 1.0, 0.5],
        'f_final_goal':          [0, 1.0, 0],
        'f_new':                 [0, 0, 1.0],
        'f_diff':                [0, 0, 1.0],
        'f_new_eq_err':          [1.0, 0, 0],
        'f_diff_eq_err':         [1.0, 0, 0],
        'f_eq_err':              [1.0, 0, 0],
        'f_settle':              [0.5, 1.0, 0],
        'f_needs_regrasp_again': [0, 1.0, 1.0],
        'f_contact':             [1.0, 0.5, 0],
    }

    def pos_transform(p):
        T = np.eye(4)
        T[0, 3] = p[0]
        T[1, 3] = p[1]
        T[2, 3] = p[2]
        return T

    for i, (costs_dict, grasp) in enumerate(zip(costs_dicts, candidate_grasps)):
        # TODO: draw one big outline box around the total cost for each solution
        z_offset = 0
        for name, cost_i in costs_dict.items():
            color_i = cost_colors[name]
            z_i = np.clip(cost_i * cost_to_meters, 1e-3, 1e3)  # ensure non-zero
            box_entity_path = f'regrasp_costs/{i}/{name}'
            log_box(box_entity_path, np.array([width, depth, z_i]),
                    pos_transform([width * i, 0, z_i / 2 + z_offset]),
                    color=color_i)
            ext = {f'grasp {gripper_idx_to_eq_name(k)}': f'{grasp.locations[k]:.2f}' for k in range(n_g)}
            ext[name] = f'{cost_i:.3f}'
            ext['total cost'] = f'{sum(costs_dict.values()):.2f}'
            ext['is_grasping'] = ' '.join([str(g) for g in grasp.is_grasping])
            rr.log_extension_components(box_entity_path, ext)
            z_offset += z_i


def pull_gripper(phy, name, loc, rope_body_indices):
    # pull the gripper that is grasping towards the position where the rope point it wants to grasp is currently
    grasp_index = grasp_location_to_indices(loc, rope_body_indices)
    grasp_world_eq = phy.m.eq(f'{name}_world')
    grasp_world_eq.active = 1
    grasp_world_eq.solref[0] = 0.06
    grasp_world_eq.data[0:3] = np.array([0, 0, GRASP_POINT_OFFSET])
    offset_world = np.zeros(3)
    x_offset = grasp_offset(grasp_index, loc, rope_body_indices)
    offset_body = np.array([x_offset, 0, 0])
    mujoco.mju_trnVecPose(offset_world, np.zeros(3), phy.d.body(grasp_index).xquat, offset_body)
    grasp_world_eq.data[3:6] = phy.d.body(grasp_index).xpos + offset_world

    # keep the other gripper where it is
    other_name = 'left' if name == 'right' else 'right'
    other_gripper_eq = phy.m.eq(f'{other_name}')
    grasp_world_eq = phy.m.eq(f'{other_name}_world')
    grasp_world_eq.active = 1
    grasp_world_eq.solref[0] = 0.06
    grasp_world_eq.data[0:3] = 0
    grasp_world_eq.data[3:6] = phy.d.body(other_gripper_eq.obj1id).xpos


def pull_rope_and_gripper_to_goal(phy, grasping_gripper_names, goal: ObjectPointGoal):
    rope_goal_eq = phy.m.eq('rope_world')
    rope_goal_eq.active = 1
    rope_goal_eq.data[0:3] = 0
    rope_goal_eq.data[3:6] = goal.goal_point

    for gripper_name in grasping_gripper_names:
        grasp_world_eq = phy.m.eq(f'{gripper_name}_world')
        grasp_world_eq.active = 1
        grasp_world_eq.solref[0] = hp['pull_gripper_to_goal_solref']
        grasp_world_eq.data[0:3] = np.array([0, 0, GRASP_POINT_OFFSET])
        grasp_world_eq.data[3:6] = goal.goal_point


def unfix_grippers(phy):
    left_world = phy.m.eq(f'left_world')
    left_world.active = 0
    right_world = phy.m.eq(f'right_world')
    right_world.active = 0


def unfix_rope(phy):
    eq = phy.m.eq(f'rope_world')
    eq.active = 0


def compute_settle_cost(phy_after, phy_before):
    settle_cost = np.linalg.norm(phy_after.d.xpos - phy_before.d.xpos)
    return settle_cost


def grasp_loc_recursive(is_grasping, i: int, grasp_locations: List):
    if i == len(is_grasping):
        yield grasp_locations
        return

    if is_grasping[i]:
        for loc in np.linspace(0.05, 0.95, 7):
            yield from grasp_loc_recursive(is_grasping, i + 1, grasp_locations + [loc])
    else:
        yield from grasp_loc_recursive(is_grasping, i + 1, grasp_locations + [0])


def grasp_weights_from_current_state(rope_body_indices, m):
    current_grasp = GraspState.from_mujoco(rope_body_indices, m)
    grasp_weights = []
    for body_idx in rope_body_indices:
        if current_grasp.is_grasping[0] and body_idx == current_grasp.indices[0]:
            grasp_weights.append(1.0)
        else:
            grasp_weights.append(0.0)
    for body_idx in rope_body_indices:
        if current_grasp.is_grasping[1] and body_idx == current_grasp.indices[1]:
            grasp_weights.append(1.0)
        else:
            grasp_weights.append(0.0)
    return np.array(grasp_weights)


def set_grasp_weights(p: Params, grasp_weights: np.ndarray):
    for i in range(int(len(grasp_weights) / 2)):
        left_w = grasp_weights[i]
        right_w = grasp_weights[int(len(grasp_weights) / 2) + i]
        p.config[f'left_w_{i}'] = float(left_w)
        p.config[f'right_w_{i}'] = float(right_w)
    p.update()


class RegraspMPC:

    def __init__(self, mppi_nu: int, pool: ThreadPoolExecutor, viz: Viz, goal, objects: Objects,
                 seed: int = 1, mov: Optional[MjMovieMaker] = None):
        self.mppi_nu = mppi_nu
        self.pool = pool
        self.viz = viz
        self.op_goal = goal
        self.objects = objects
        self.rope_body_indices = self.objects.rope.body_indices
        self.n_g = hp['n_g']
        self.mov = mov
        self.is_gasping_rng = np.random.RandomState(0)

        self.mppi = MujocoMPPI(pool=self.pool, nu=mppi_nu, seed=seed, noise_sigma=np.deg2rad(2), horizon=hp['horizon'],
                               temp=hp['temp'])
        self.state_history = Buffer(hp['state_history_size'])
        self.max_dq = None
        self.reset_trap_detection()

    def reset_trap_detection(self):
        self.state_history.reset()
        self.max_dq = 0

    def run_combined(self, phy):
        initial_grasp_weights = grasp_weights_from_current_state(self.rope_body_indices, phy.m)
        set_grasp_weights(self.viz.p, initial_grasp_weights)

        sub_time_s = hp['move_sub_time_s']
        num_samples = hp['num_samples']

        itr = 0
        self.mppi.reset()
        warmstarting = 0.0  # use a float here then cast to int when iterating
        last_costs = None
        while True:
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown")

            self.op_goal.viz_goal(phy)
            if self.op_goal.satisfied(phy):
                return Result(Status.SUCCESS, f"Goal reached! ({itr} iters)", None)

            self.viz.viz(phy)

            # these weights are from 0 to 1 and are used to weight the cost terms for each potential grasp
            grasp_w = self.op_goal.get_grasp_weights()

            for i in range(int(warmstarting)):
                command = self.mppi.command(phy, self.op_goal.get_results, self.op_goal.cost, sub_time_s, num_samples)
                self.mppi_viz(self.mppi, self.op_goal, phy, command, sub_time_s)
            command = self.mppi.command(phy, self.op_goal.get_results, self.op_goal.cost, sub_time_s, num_samples)
            self.mppi_viz(self.mppi, self.op_goal, phy, command, sub_time_s)

            # NOTE: disable because this wasn't a well thought out idea and didn't help that much
            # adjust warmstarting based on how much the costs have changed
            # if last_costs is not None:
            #     t_stat, p_value = ttest_ind(last_costs[:, 1:].mean(-1), self.mppi.costs[:, :-1].mean(-1))
            #     p_value = float(p_value)
            #     warmstarting += (0.1 - p_value) * 1  # increase warmstarting if the costs are different
            #     warmstarting = max(min(warmstarting, hp['warmstart']), 0)

            # If the weight is high and the cost is low enough, commit to actually making the grasp
            for gripper_i in range(self.n_g):
                gripper_name = gripper_idx_to_eq_name(gripper_i)
                current_eq = phy.m.eq(gripper_name)
                for body_j, body_idx in enumerate(self.rope_body_indices):
                    grasp_w_ij = grasp_w[gripper_i, body_j]
                    # FIXME: is mean the right thing to do here?
                    per_grasp_cost = np.mean(self.op_goal.grasp_costs[gripper_i, body_j])
                    near = per_grasp_cost < hp['cost_activation_thresh']
                    already_grasping = (bool(current_eq.active) and bool(current_eq.obj2id == body_idx))
                    if grasp_w_ij > hp['weight_activation_thresh'] and near and not already_grasping:
                        # Todo: how can we have continuous grasp locations instead of discretizing?
                        loc = grasp_indices_to_locations(self.rope_body_indices, body_idx)
                        # Note: in the real world we will need to run some closed-loop grasping controller
                        #   but the gripper should already very close to the grasp location
                        print(f"Activating grasp {gripper_i} on body {body_j}")
                        activate_grasp(phy, gripper_name, loc, self.rope_body_indices)
                        # Reset since cost landscape has changed a lot
                        self.mppi.reset()

            # conversely, if the weight is low and the grasp is currently being made, release the grasp
            for gripper_i in range(self.n_g):
                gripper_name = gripper_idx_to_eq_name(gripper_i)
                current_eq = phy.m.eq(gripper_name)
                for body_j, body_idx in enumerate(self.rope_body_indices):
                    grasp_w_ij = grasp_w[gripper_i, body_j]
                    is_grasping = bool(current_eq.active) and current_eq.obj2id == body_idx
                    if grasp_w_ij < hp['weight_deactivation_thresh'] and is_grasping:
                        print(f"Deactivating grasp {gripper_i} on body {body_j}")
                        current_eq.active[0] = 0
                        # Reset since cost landscape has changed a lot
                        self.mppi.reset()

            control_step(phy, command, sub_time_s)

            self.viz.viz(phy, False)
            if self.mov:
                self.mov.render(phy.d)

            self.mppi.roll()

            if itr > hp['max_move_to_goal_iters']:
                return Result(Status.FAILED, f"Gave up after {itr} iters", None)

            last_costs = self.mppi.costs

            itr += 1

    def run(self, phy):

        # copy model and data since each solution should be different/independent
        num_samples = hp['regrasp_n_samples']

        regrasp_goal = RegraspGoal(self.op_goal, hp['grasp_goal_radius'], self.objects, self.viz)
        initial_exploration_weight = 0.02
        exploration_weight = initial_exploration_weight

        # TODO: seed properly
        mppi = RegraspMPPI(pool=self.pool, nu=self.mppi_nu, seed=0, horizon=hp['regrasp_horizon'],
                           noise_sigma=np.deg2rad(5),
                           n_g=self.n_g, rope_body_indices=self.rope_body_indices, temp=hp['regrasp_temp'])
        mppi.reset()
        self.reset_trap_detection()

        itr = 0
        max_iters = 500
        sub_time_s = hp['plan_sub_time_s']
        warmstart_count = 0
        self.viz.viz(phy)
        while True:
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown")

            if itr > max_iters:
                return False

            regrasp_goal.viz_goal(phy)
            if regrasp_goal.satisfied(phy):
                print("Goal reached!")
                return True

            needs_regrasp = self.check_needs_regrasp(phy.d)
            if needs_regrasp:
                print("trap detected!")
                self.reset_trap_detection()
                exploration_weight = np.clip(exploration_weight * 4, 0, 100)

            while warmstart_count < hp['warmstart']:
                command = mppi.command(phy, regrasp_goal, sub_time_s, num_samples, exploration_weight, viz=self.viz)
                self.mppi_viz(mppi, regrasp_goal, phy, None, sub_time_s)
                warmstart_count += 1

            command = mppi.command(phy, regrasp_goal, sub_time_s, num_samples, exploration_weight, viz=self.viz)
            self.mppi_viz(mppi, regrasp_goal, phy, None, sub_time_s)

            control_step(phy, command, sub_time_s)
            self.viz.viz(phy)

            if self.mov:
                self.mov.render(phy.d)

            left_tool_pos, right_tool_pos = get_tool_positions(phy)
            did_new_grasp = do_grasps_if_close(phy, left_tool_pos, right_tool_pos, self.rope_body_indices)
            if did_new_grasp:
                print("New grasp!")
                self.reset_trap_detection()
                warmstart_count = 0
                mppi.reset()
                exploration_weight = initial_exploration_weight
            did_release = release_dynamics(phy)
            if did_release:
                print("Released!")
                self.reset_trap_detection()
                warmstart_count = 0
                mppi.reset()
                exploration_weight = initial_exploration_weight

            mppi.roll()

            itr += 1

    def run_old(self, phy):
        for self.itr in range(hp['iters']):
            if rospy.is_shutdown():
                self.close()
                raise RuntimeError("ROS shutdown")

            logger.info(Fore.BLUE + f"Moving to goal" + Fore.RESET)
            move_result = self.move_to_goal(phy, hp['max_move_to_goal_iters'], is_planning=False,
                                            sub_time_s=hp['move_sub_time_s'], num_samples=hp['num_samples'])

            if move_result.status in [Status.SUCCESS]:
                self.close()
                return move_result

            while True:
                # NOTE: currently testing what happens if we let it regrasp if a grasp fails
                logger.info(Fore.BLUE + f"Computing New Grasp" + Fore.RESET)
                save_data_and_eq(phy)
                grasp = self.compute_new_grasp_mppi(phy)
                logger.info(Fore.BLUE + f"Grasping {grasp}" + Fore.RESET)
                grasp_result = self.do_multi_gripper_regrasp(phy, grasp, max_iters=hp['max_grasp_iters'],
                                                             is_planning=False, sub_time_s=hp['grasp_sub_time_s'],
                                                             stop_if_failed=True, num_samples=hp['num_samples'])
                if grasp_result.status == Status.SUCCESS:
                    break

    def check_needs_regrasp(self, data):
        latest_q = self.get_q_for_trap_detection(data)
        self.state_history.insert(latest_q)
        qs = np.array(self.state_history.data)
        if len(qs) < 2:
            dq = 0
        else:
            # distance between the newest and oldest q in the buffer
            # the mean takes the average across joints.
            dq = (np.abs(qs[-1] - qs[0]) / len(self.state_history)).mean()
        self.max_dq = max(self.max_dq, dq)
        has_not_moved = dq < hp['frac_max_dq'] * self.max_dq
        needs_regrasp = self.state_history.full() and has_not_moved
        rr.log_scalar('needs_regrasp/dq', dq, color=[0, 255, 0])
        rr.log_scalar('needs_regrasp/max_dq', self.max_dq, color=[0, 0, 255])
        rr.log_scalar('needs_regrasp/dq_threshold', self.max_dq * hp['frac_max_dq'], color=[255, 0, 0])
        return needs_regrasp

    def get_q_for_trap_detection(self, data):
        return np.concatenate((hp['q_joint_weight'] * data.qpos[self.objects.rope.qpos_indices],
                               data.qpos[self.objects.val.qpos_indices]))

    def exhaustive_new_grasp_search(self, phy: Physics):
        all_costs_dicts = []
        all_grasps = []
        grasp0 = GraspState.from_mujoco(self.rope_body_indices, phy.m)
        for is_grasping in itertools.product([0, 1], repeat=self.n_g):
            is_grasping = np.array(is_grasping)
            for grasp_locations in grasp_loc_recursive(is_grasping, 0, []):
                grasp_locations = np.array(grasp_locations)
                grasp = GraspState(self.rope_body_indices, grasp_locations, is_grasping)
                costs_dict, status = self.score_grasp_location(phy, grasp0, grasp)
                total_cost = sum(costs_dict.values())
                print(grasp, total_cost)
                all_costs_dicts.append(costs_dict)
                all_grasps.append(grasp)
                viz_regrasp_solutions_and_costs(all_costs_dicts, all_grasps)

    def compute_new_grasp_cma(self, phy: Physics):
        grasp0 = GraspState.from_mujoco(self.rope_body_indices, phy.m)

        all_costs_dicts = []
        all_grasps = []

        # Run CMA-ES to find the best grasp.
        # I've experimenting with different objective functions
        # TODO: try a new categorical formulation of the binary grasp space, where we optimize over (4) logits
        #  representing the (4) possible grasp states, and then sample from the distribution to evaluate cost.
        #  We would still then have the (2) continuous [0-1] variables for the grasp locations.
        # x0 is a vector of the form [p(is_grasping), grasp_locations]
        # and is initialized with a bias towards inverting the current grasp and grasping in the middle
        x0 = np.concatenate([(1 - grasp0.is_grasping) * 0.8, np.ones_like(grasp0.locations) * 0.5])
        es = cma.CMAEvolutionStrategy(x0=x0, sigma0=hp['cma_sigma'], inopts=hp['cma_opts'])
        # CMA keeps track of this internally, but we do it ourselves since we don't want to have to re-sample
        # the is_grasping variables, which could cause a different binary is_grasping state than the one we evaluated
        grasp_best = None
        fbest = None
        while not es.stop():
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown")

            candidates_xs = es.ask()
            candidate_costs = []
            for x_i in candidates_xs:
                is_grasping = self.sample_is_grasping(x_i[:self.n_g])
                grasp_locations = x_i[self.n_g:]
                grasp = GraspState(self.rope_body_indices, grasp_locations, is_grasping)
                logger.info(f"Score {grasp}")

                from time import perf_counter
                t0 = perf_counter()
                costs_dict, status = self.score_grasp_location(phy, grasp0, grasp)
                total_cost = sum(costs_dict.values())
                candidate_costs.append(total_cost)
                logger.info(f'{grasp=} {total_cost=:.3f} dt={perf_counter() - t0:.2f}')
                if status == Status.SUCCESS:
                    # Shortcut to return any trajectory that reaches the goal
                    logger.info(Fore.GREEN + f"Best grasp: {grasp=}, {total_cost=:.2f}" + Fore.RESET)
                    return grasp

                all_grasps.append(grasp)
                all_costs_dicts.append(costs_dict)

                # Visualize!
                viz_regrasp_solutions_and_costs(all_costs_dicts, all_grasps)

                # track the best
                if fbest is None or total_cost < fbest:
                    fbest = total_cost
                    grasp_best = grasp
                    logger.info(Fore.GREEN + f"New best grasp: {grasp_best=}, {fbest=:.2f}" + Fore.RESET)

            es.tell(candidates_xs, candidate_costs)

        print(es.result_pretty())
        logger.info(Fore.GREEN + f"Best grasp: {grasp_best=}, {fbest=:.2f}" + Fore.RESET)
        return grasp_best

    def cma_result_to_grasp(self, result):
        best_is_grasping = self.sample_is_grasping(result.xbest[:self.n_g])
        best_grasp_locations = result.xbest[self.n_g:]
        grasp_best = GraspState(self.rope_body_indices, best_grasp_locations, best_is_grasping)
        return grasp_best

    def score_grasp_location(self, parent_phy: Physics, grasp0: GraspState, grasp: GraspState):
        # copy model and data since each solution should be different/independent
        phy = parent_phy.copy_all()

        # Perhaps we could get rid of these shortcuts by implementing a more general shortcut system
        # that stops cost evaluation when it becomes so high that it doesn't contribute to the best solution.
        if np.all(np.logical_not(grasp.is_grasping)):
            # I'm only including all terms in the dict because otherwise the viz in rerun looks bad
            return {
                'f_all_0':               100,
                'f_is_same':             0,
                'f_eq_err':              0,
                'f_contact':             0,
                'f_settle':              0,
                'f_needs_regrasp_again': 0,
            }, Status.FAILED
        if grasp0 == grasp:
            return {
                'f_is_same':             100,
                'f_all_0':               0,
                'f_eq_err':              0,
                'f_contact':             0,
                'f_settle':              0,
                'f_needs_regrasp_again': 0,
            }, Status.FAILED

        # Instead of doing a full grasp simulation, activate eq constraints between the gripper and the world
        # position where the gripper should be grasping, and the gripper and the rope body.
        # The cost would then consist of the eq errors, contact cost, and settling cost.
        # We can do the same for replacing move_to_goal, by activating an eq constraint between the rope body and the
        # world at the position of the goal, then letting it settle. The cost would again be eq errors, contact cost,
        # and settling cost.

        settle_steps = hp['plan_settle_steps']
        is_new = grasp0.is_new(grasp)
        is_diff = grasp0.is_diff(grasp)
        needs_release = grasp0.needs_release(grasp)
        f_eq_errs = []
        f_settles = []
        contact_costs = []
        grasping_gripper_names = []
        for i in range(self.n_g):
            if grasp.is_grasping[i]:
                grasping_gripper_names.append(gripper_idx_to_eq_name(i))

        def _grasp(name, loc):
            pull_gripper(phy, name, loc, self.rope_body_indices)
            phy_before = phy.copy_data()
            ctrl = np.zeros(phy.m.nu)
            ctrl[self.objects.gripper_ctrl_indices] = 0.2
            settle(phy, hp['plan_sub_time_s'], self.viz, True, hp['plan_settle_steps'], ctrl=ctrl)

            activate_grasp(phy, name, loc, self.rope_body_indices)
            # Let the newly activate eq's settle, and command the grippers to close
            ctrl = np.zeros(phy.m.nu)
            ctrl[self.objects.gripper_ctrl_indices] = -0.5
            settle_results = settle(phy, hp['plan_sub_time_s'], self.viz, True, hp['plan_settle_steps'], ctrl=ctrl)
            # TODO: penalize contact cost during the settle?

            phy_after = phy.copy_data()
            f_settles.append(compute_settle_cost(phy_after, phy_before))
            f_eq_errs.append(compute_eq_errors(phy))
            contact_costs.append(get_contact_cost(phy, self.objects))

            # undo the constraints keeping the grippers fixed in the world
            # we only do that in order to get a more realistic estimate of
            # the cost/feasibility of the grasp
            unfix_grippers(phy)

        def _policy(_phy):
            ctrl = np.zeros(_phy.m.nu)
            # use the jacobian to compute the joint velocities which would move the gripper in the direction from
            # rope to goal
            rope_pos = _phy.d.xpos[self.rope_body_indices[self.op_goal.body_idx]]
            goal_pos = self.op_goal.goal_point
            gripper_dir = goal_pos - rope_pos

            for i in range(self.n_g):
                if grasp.is_grasping[i]:
                    gripper_name = gripper_idx_to_eq_name(i)

        def _move_to_goal():
            self.op_goal.viz_goal(phy)
            phy_before = phy.copy_data()
            settle_result = settle(phy, hp['plan_sub_time_s'], self.viz, True, hp['plan_settle_steps'],
                                   get_result_func=self.op_goal.get_results, policy=_policy)
            settle_results = expand_result(settle_result)
            f_goal_cost = self.op_goal.point_dist_cost(settle_results)[0, -1]
            f_goal_cost = f_goal_cost * self.mppi.gamma * hp['f_goal_weight']

            # undo the constraints pulling the rope towards the goal
            unfix_rope(phy)

            final_settle_result = settle(phy, hp['plan_sub_time_s'], self.viz, True, hp['plan_settle_steps'],
                                         get_result_func=self.op_goal.get_results)
            final_settle_results = expand_result(final_settle_result)
            final_goal_cost = self.op_goal.point_dist_cost(final_settle_results)[0, -1]
            final_goal_cost = final_goal_cost * self.mppi.gamma * hp['f_final_goal_weight']
            phy_after = phy.copy_data()
            f_settles.append(compute_settle_cost(phy_after, phy_before))
            f_eq_errs.append(compute_eq_errors(phy))

            return f_goal_cost, final_goal_cost

        for gripper_idx in range(self.n_g):
            if is_new[gripper_idx]:
                name = gripper_idx_to_eq_name(gripper_idx)
                loc = grasp.locations[gripper_idx]
                _grasp(name, loc)
        # deactivate
        for gripper_idx in range(self.n_g):
            if is_diff[gripper_idx] or needs_release[gripper_idx]:
                name = gripper_idx_to_eq_name(gripper_idx)
                deactivate_eq(phy.m, name)
                phy_before = phy.copy_data()
                settle(phy, hp['plan_sub_time_s'], self.viz, True, settle_steps, mov=None)
                phy_after = phy.copy_data()
                settle_cost = compute_settle_cost(phy_after, phy_before)
                f_settles.append(settle_cost)
                contact_costs.append(get_contact_cost(phy, self.objects))
        for gripper_idx in range(self.n_g):
            if is_diff[gripper_idx]:
                name = gripper_idx_to_eq_name(gripper_idx)
                loc = grasp.locations[gripper_idx]
                _grasp(name, loc)

        # Also penalize errors after all the grasps have changed
        final_eq_err = compute_eq_errors(phy)
        f_eq_errs.append(final_eq_err)

        f_goal, f_final_goal = _move_to_goal()

        f_eq_err = sum(f_eq_errs) * hp['f_eq_err_weight']
        f_settle = sum(f_settles) * hp['f_settle_weight']
        f_contact = sum(contact_costs) * hp['f_contact_weight']
        return {
            'f_goal':       f_goal,
            'f_final_goal': f_final_goal,
            'f_eq_err':     f_eq_err,
            'f_contact':    f_contact,
            'f_settle':     f_settle,
            'f_all_0':      0,
            'f_is_same':    0,
        }, Status.FAILED

    def do_multi_gripper_regrasp(self, phy, grasp, max_iters, is_planning: bool, sub_time_s: float, num_samples: int,
                                 stop_if_failed=False):
        settle_steps = hp['plan_settle_steps'] if is_planning else hp['settle_steps']
        grasp0 = GraspState.from_mujoco(self.rope_body_indices, phy.m)
        is_new = grasp0.is_new(grasp)
        is_diff = grasp0.is_diff(grasp)
        needs_release = grasp0.needs_release(grasp)
        # For each gripper, if it needs a _new_ grasp, run MPPI to try to find a grasp and add the cost
        # NOTE: technically "order" of which gripper we consider first matters, we should in theory try all
        f_news = []
        f_new_eq_errs = []
        f_diffs = []
        f_diff_eq_errs = []
        for gripper_idx in range(self.n_g):
            if is_new[gripper_idx]:
                # plan the grasp
                f_new_result = self.do_single_gripper_grasp(phy, grasp, gripper_idx, max_iters, is_planning, sub_time_s,
                                                            hp['num_samples'])
                if f_new_result.status in [Status.FAILED] and stop_if_failed:
                    return f_new_result
                f_news.extend(f_new_result.cost)
                # activate new grasp
                change_eq(phy.m, gripper_idx_to_eq_name(gripper_idx), grasp.locations[gripper_idx],
                          self.rope_body_indices)
                settle(phy, sub_time_s, self.viz, is_planning, settle_steps, mov=self.mov)
                # add error for the grasp changing the state a lot, or for the eq constraint not being met
                # mm means 'mismatch'
                f_new_eq_err_i = compute_eq_errors(phy)
                f_new_eq_errs.append(f_new_eq_err_i)
        # deactivate
        for gripper_idx in range(self.n_g):
            if is_diff[gripper_idx] or needs_release[gripper_idx]:
                deactivate_eq(phy.m, gripper_idx_to_eq_name(gripper_idx))
                settle(phy, sub_time_s, self.viz, is_planning, settle_steps, mov=self.mov)
        # For each gripper, if it needs a different grasp, run MPPI to try to find a grasp and add the cost
        for gripper_idx in range(self.n_g):
            if is_diff[gripper_idx]:
                f_diff_result = self.do_single_gripper_grasp(phy, grasp, gripper_idx, max_iters, is_planning,
                                                             sub_time_s, num_samples)
                if f_diff_result.status in [Status.FAILED] and stop_if_failed:
                    return f_diff_result
                f_diffs.extend(f_diff_result.cost)
                change_eq(phy.m, gripper_idx_to_eq_name(gripper_idx), grasp.locations[gripper_idx],
                          self.rope_body_indices)
                settle(phy, sub_time_s, self.viz, is_planning, settle_steps, mov=self.mov)
                f_diff_eq_err_i = compute_eq_errors(phy)
                f_diff_eq_errs.append(f_diff_eq_err_i)

        return Result(Status.SUCCESS, cost=(f_news, f_new_eq_errs, f_diffs, f_diff_eq_errs))

    def do_single_gripper_grasp(self, phy, grasp: GraspState, gripper_idx: int, max_iters: int, is_planning: bool,
                                sub_time_s: float, num_samples: int):
        offset = grasp.offsets[gripper_idx]
        rope_body_to_grasp = phy.m.body(grasp.indices[gripper_idx])

        self.mppi.reset()
        self.reset_trap_detection()
        self.viz.viz(phy, is_planning)

        warmstart_count = 0
        execution_costs = []
        grasp_iter = 0
        grasp_goal = GraspRopeGoal(body_id_to_grasp=rope_body_to_grasp.id,
                                   goal_radius=hp['grasp_goal_radius'],
                                   offset=offset,
                                   gripper_idx=gripper_idx,
                                   viz=self.viz,
                                   objects=self.objects)
        while True:
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown")

            grasp_goal.viz_goal(phy)
            if grasp_goal.satisfied(phy):
                execution_costs.append(self.mppi.get_min_terminal_cost())
                return Result(Status.SUCCESS, f"Grasp successful ({grasp_iter} iters)", execution_costs)

            while warmstart_count < hp['warmstart']:
                command = self.mppi.command(phy, grasp_goal.get_results, grasp_goal.cost, sub_time_s,
                                            num_samples)
                self.mppi_viz(self.mppi, grasp_goal, phy, command, sub_time_s)
                warmstart_count += 1
            command = self.mppi.command(phy, grasp_goal.get_results, grasp_goal.cost, sub_time_s, num_samples)
            self.mppi_viz(self.mppi, grasp_goal, phy, command, sub_time_s)

            control_step(phy, command, sub_time_s)
            self.viz.viz(phy, is_planning)
            if not is_planning and self.mov:
                self.mov.render(phy.d)
            self.mppi.roll()

            if grasp_iter > max_iters:
                execution_costs.append(self.mppi.get_min_terminal_cost())
                return Result(Status.FAILED, f"Failed to grasp after {grasp_iter} iters", execution_costs)

            grasp_iter += 1
            execution_costs.append(self.mppi.get_first_step_cost() * hp['running_cost_weight'])

    def combined_goal_and_grasping(self, phy, max_iters, is_planning: bool, sub_time_s: float, num_samples: int):
        pass

    def move_to_goal(self, phy, max_iters, is_planning: bool, sub_time_s: float, num_samples: int):
        self.mppi.reset()
        self.reset_trap_detection()
        self.viz.viz(phy, is_planning)
        warmstart_count = 0
        execution_costs = []
        move_iter = 0

        while True:
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown")

            self.op_goal.viz_goal(phy)
            if self.op_goal.satisfied(phy):
                execution_costs.append(self.mppi.get_min_terminal_cost())
                return Result(Status.SUCCESS, f"Goal reached! ({move_iter} iters)", execution_costs)

            while warmstart_count < hp['warmstart']:
                command = self.mppi.command(phy, self.op_goal.get_results, self.op_goal.cost, sub_time_s, num_samples)
                self.mppi_viz(self.mppi, self.op_goal, phy, command, sub_time_s)
                warmstart_count += 1
            command = self.mppi.command(phy, self.op_goal.get_results, self.op_goal.cost, sub_time_s, num_samples)
            self.mppi_viz(self.mppi, self.op_goal, phy, command, sub_time_s)

            control_step(phy, command, sub_time_s)
            self.viz.viz(phy, is_planning)
            if not is_planning and self.mov:
                self.mov.render(phy.d)

            self.mppi.roll()

            needs_regrasp = self.check_needs_regrasp(phy.d)
            if needs_regrasp:
                execution_costs.append(self.mppi.get_min_terminal_cost())
                return Result(Status.REGRASP, "Needs regrasp", execution_costs)

            if move_iter > max_iters:
                execution_costs.append(self.mppi.get_min_terminal_cost())
                return Result(Status.FAILED, f"Gave up after {move_iter} iters", execution_costs)

            execution_costs.append(self.mppi.get_first_step_cost() * hp['running_cost_weight'])
            move_iter += 1

    def mppi_viz(self, mppi, goal, phy, command, sub_time_s):
        if not self.viz.p.mppi_rollouts:
            return

        sorted_traj_indices = np.argsort(mppi.cost)

        i = None
        num_samples = mppi.cost.shape[0]
        for i in range(min(num_samples, 10)):
            sorted_traj_idx = sorted_traj_indices[i]
            cost_normalized = mppi.cost_normalized[sorted_traj_idx]
            c = cm.RdYlGn(1 - cost_normalized)
            result_i = mppi.rollout_results[:, sorted_traj_idx]
            goal.viz_result(result_i, i, color=c, scale=0.002)
            rospy.sleep(0.01)  # needed otherwise messages get dropped :( I hate ROS...

        if command is not None:
            cmd_rollout_results = rollout(phy.copy_data(), command[None], sub_time_s, get_result_func=goal.get_results)
            goal.viz_result(cmd_rollout_results, i, color='b', scale=0.004)

    def close(self):
        if self.mov is not None:
            self.mov.close()

    def sample_is_grasping(self, param):
        r = self.is_gasping_rng.uniform(size=self.n_g)
        is_grasping = (r < param).astype(int)
        return is_grasping
