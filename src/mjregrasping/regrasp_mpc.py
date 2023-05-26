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

import rospy
from mjregrasping.body_with_children import Objects
from mjregrasping.buffer import Buffer
from mjregrasping.change_grasp_eq import change_eq
from mjregrasping.goals import ObjectPointGoal, GraspRopeGoal, get_contact_cost
from mjregrasping.grasp_state import GraspState, grasp_location_to_indices, grasp_offset
from mjregrasping.grasping import deactivate_eq, compute_eq_errors
from mjregrasping.mjsaver import save_data_and_eq
from mjregrasping.movie import MjMovieMaker
from mjregrasping.mujoco_mppi import MujocoMPPI
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import log_box
from mjregrasping.rollout import control_step, rollout
from mjregrasping.scenes import settle
from mjregrasping.viz import Viz

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


def gripper_idx_to_eq_name(gripper_idx):
    return 'left' if gripper_idx == 0 else 'right'


def viz_regrasp_solutions_and_costs(costs_dicts: List[Dict], candidate_grasps: List[GraspState]):
    # histograms
    n_g = 2
    width = 0.1
    cost_to_meters = 0.1
    depth = 0.05
    cost_colors = {
        'f_is_same':             [255, 0, 255],
        'f_all_0':               [255, 0, 255],
        'f_goal':                [0, 255, 0],
        'f_new':                 [0, 0, 255],
        'f_diff':                [0, 0, 255],
        'f_new_eq_err':          [255, 0, 0],
        'f_diff_eq_err':         [255, 0, 0],
        'f_eq_err':              [255, 0, 0],
        'f_settle':              [255, 255, 0],
        'f_needs_regrasp_again': [0, 255, 255],
        'f_contact':             [128, 255, 128],
        'f_is_diff':             [255, 128, 0],
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
            ext = {f'grasp {gripper_idx_to_eq_name(k)}': grasp.locations[k] for k in range(n_g)}
            ext[name] = f'{cost_i:.3f}'
            ext['total cost'] = sum(costs_dict.values())
            ext['is_grasping'] = ' '.join([str(g) for g in grasp.is_grasping])
            rr.log_extension_components(box_entity_path, ext)
            z_offset += z_i


def pull_gripper(phy, name, loc, rope_body_indices):
    grasp_index = grasp_location_to_indices(loc, rope_body_indices)
    x_offset = grasp_offset(grasp_index, loc, rope_body_indices)
    # Round the x_offset here so that the eqs are not too sensitive to small changes in the x_offset.
    # I was noticing that I got different cost when comparing `regrasp` and `untangle` and it turns out it was caused
    # by the x_offset being slightly different, since in one case the loc is sampled from code and in the other
    # it's typed in manually (to 3 decimal places).
    x_offset = round(x_offset, 2)
    grasp_eq = phy.m.eq(name)
    grasp_eq.obj2id = grasp_index
    grasp_eq.active = 1
    offset_body = np.array([x_offset, 0, 0])
    grasp_eq.data[3:6] = offset_body

    # pull the gripper that is grasping towards the position where the rope point it wants to grasp is currenty
    grasp_index = grasp_location_to_indices(loc, rope_body_indices)
    grasp_world_eq = phy.m.eq(f'{name}_world')
    grasp_world_eq.active = 1
    grasp_world_eq.data[0:3] = np.array([0, 0, 0.16])
    offset_world = np.zeros(3)
    mujoco.mju_trnVecPose(offset_world, np.zeros(3), phy.d.body(grasp_index).xquat, offset_body)
    grasp_world_eq.data[3:6] = phy.d.body(grasp_index).xpos + offset_world

    # keep the other gripper where it is
    other_name = 'left' if name == 'right' else 'right'
    other_gripper_eq = phy.m.eq(f'{other_name}')
    grasp_world_eq = phy.m.eq(f'{other_name}_world')
    grasp_world_eq.active = 1
    grasp_world_eq.data[0:3] = 0
    grasp_world_eq.data[3:6] = phy.d.body(other_gripper_eq.obj1id).xpos


def unfix_rope(phy):
    left_world = phy.m.eq(f'left_world')
    left_world.active = 0
    right_world = phy.m.eq(f'right_world')
    right_world.active = 0


def compute_settle_cost(phy_after, phy_before):
    settle_cost = np.linalg.norm(phy_after.d.xpos - phy_before.d.xpos)
    return settle_cost


class RegraspMPC:

    def __init__(self, mppi_nu: int, pool: ThreadPoolExecutor, viz: Viz, goal: ObjectPointGoal, objects: Objects,
                 seed: int = 1, mov: Optional[MjMovieMaker] = None):
        self.max_dq = None
        self.viz = viz
        self.goal = goal
        self.objects = objects
        self.rope_body_indices = np.array(self.objects.rope.body_indices)
        self.n_g = 2
        self.mov = mov
        self.is_gasping_rng = np.random.RandomState(0)

        self.mppi = MujocoMPPI(pool=pool, nu=mppi_nu, seed=seed, noise_sigma=np.deg2rad(4), horizon=hp['horizon'],
                               lambda_=hp['lambda'])
        self.state_history = Buffer(12)
        self.reset_trap_detection()

        self.regrasp_idx = 0

    def reset_trap_detection(self):
        self.state_history.reset()
        self.max_dq = 0

    def run(self, phy):
        for self.iter in range(hp['iters']):
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
                grasp = self.compute_new_grasp(phy)
                logger.info(Fore.BLUE + f"Grasping {grasp}" + Fore.RESET)
                grasp_result = self.do_multi_gripper_regrasp(phy, grasp, max_iters=hp['max_grasp_iters'],
                                                             is_planning=False, sub_time_s=hp['grasp_sub_time_s'],
                                                             stop_if_failed=True, num_samples=hp['num_samples'])
                if grasp_result.status == Status.SUCCESS:
                    break

            self.regrasp_idx += 1

    def check_needs_regrasp(self, data):
        latest_q = data.qpos[self.objects.rope.qpos_indices]
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

    def compute_new_grasp(self, phy: Physics):
        grasp0 = GraspState.from_mujoco(self.rope_body_indices, phy.m)

        all_costs_dicts = []
        all_grasps = []

        # Run CMA-ES to find the best grasp.
        # The objective function consists of solving a series of planning problems with MPPI.
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
                'f_all_0':               10,
                'f_is_same':             0,
                'f_goal':                0,
                'f_eq_err':              0,
                'f_contact':             0,
                'f_settle':              0,
                'f_needs_regrasp_again': 0,
                'f_is_diff':             0,
            }, Status.FAILED
        if grasp0 == grasp:
            return {
                'f_is_same':             10,
                'f_all_0':               0,
                'f_goal':                0,
                'f_eq_err':              0,
                'f_contact':             0,
                'f_settle':              0,
                'f_needs_regrasp_again': 0,
                'f_is_diff':             0,
            }, Status.FAILED

        # what about instead of doing a full grasp simulation,
        # we just activate the eq constraints and then try to move to the goal?
        # The cost would then consist of the eq errors and the goal error?
        settle_steps = hp['plan_settle_steps']
        is_new = grasp0.is_new(grasp)
        is_diff = grasp0.is_diff(grasp)
        needs_release = grasp0.needs_release(grasp)
        f_eq_errs = []
        f_settles = []
        contact_costs = []

        def _grasp(name, loc):
            pull_gripper(phy, name, loc, self.rope_body_indices)
            phy_before = phy.copy_data()
            # Let the newly activate eq's settle, and command the grippers to close
            ctrl = np.zeros(phy.m.nu)
            ctrl[self.objects.gripper_ctrl_indices] = -0.5
            for _ in range(settle_steps):
                self.viz.viz(phy, is_planning=True)
                control_step(phy, ctrl, sub_time_s=hp['plan_sub_time_s'])

            phy_after = phy.copy_data()
            settle_cost = compute_settle_cost(phy_after, phy_before)
            f_settles.append(settle_cost)
            eq_err = compute_eq_errors(phy)
            f_eq_errs.append(eq_err)

            # undo the constraints keeping the grippers fixed in the world
            # we only do that in order to get a more realistic estimate of
            # the cost/feasibility of the grasp
            unfix_rope(phy)

        f_is_diff = 0
        for gripper_idx in range(self.n_g):
            if is_new[gripper_idx]:
                name = gripper_idx_to_eq_name(gripper_idx)
                loc = grasp.locations[gripper_idx]
                _grasp(name, loc)
                contact_costs.append(get_contact_cost(phy, self.objects))
        # deactivate
        for gripper_idx in range(self.n_g):
            if is_diff[gripper_idx] or needs_release[gripper_idx]:
                name = gripper_idx_to_eq_name(gripper_idx)
                deactivate_eq(phy.m, name)
                phy_before = phy.copy_data()
                settle(phy, hp['plan_sub_time_s'], self.viz, True, settle_steps)
                phy_after = phy.copy_data()
                settle_cost = compute_settle_cost(phy_after, phy_before)
                f_settles.append(settle_cost)
                contact_costs.append(get_contact_cost(phy, self.objects))
        for gripper_idx in range(self.n_g):
            if is_diff[gripper_idx]:
                name = gripper_idx_to_eq_name(gripper_idx)
                loc = grasp.locations[gripper_idx]
                _grasp(name, loc)
                contact_costs.append(get_contact_cost(phy, self.objects))

        # Also penalize errors after all the grasps have changed
        final_eq_err = compute_eq_errors(phy)
        f_eq_errs.append(final_eq_err)

        move_result = self.move_to_goal(phy, hp['max_plan_to_goal_iters'], is_planning=True,
                                        sub_time_s=hp['plan_sub_time_s'], num_samples=hp['num_samples_when_scoring'])
        f_goal = sum(move_result.cost) * hp['f_goal_weight']
        f_eq_err = sum(f_eq_errs) * hp['f_eq_err_weight']
        f_settle = sum(f_settles) * hp['f_settle_weight']
        f_contact = sum(contact_costs) * hp['f_contact_weight']
        return {
            'f_goal':                f_goal,
            'f_eq_err':              f_eq_err,
            'f_contact':             f_contact,
            'f_settle':              f_settle,
            'f_needs_regrasp_again': hp['needs_regrasp_again'] if move_result.status == Status.REGRASP else 0,
            'f_all_0':               0,
            'f_is_same':             0,
            'f_is_diff':             f_is_diff,
        }, move_result.status

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
                settle(phy, sub_time_s, self.viz, is_planning, settle_steps)
                # add error for the grasp changing the state a lot, or for the eq constraint not being met
                # mm means 'mismatch'
                f_new_eq_err_i = compute_eq_errors(phy)
                f_new_eq_errs.append(f_new_eq_err_i)
        # deactivate
        for gripper_idx in range(self.n_g):
            if is_diff[gripper_idx] or needs_release[gripper_idx]:
                deactivate_eq(phy.m, gripper_idx_to_eq_name(gripper_idx))
                settle(phy, sub_time_s, self.viz, is_planning, settle_steps)
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
                settle(phy, sub_time_s, self.viz, is_planning, settle_steps)
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
                                   goal_radius=0.015,
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
                self.mppi_viz(grasp_goal, phy, command, sub_time_s)
                warmstart_count += 1
            command = self.mppi.command(phy, grasp_goal.get_results, grasp_goal.cost, sub_time_s, num_samples)
            self.mppi_viz(grasp_goal, phy, command, sub_time_s)

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

            self.goal.viz_goal(phy)
            if self.goal.satisfied(phy):
                execution_costs.append(self.mppi.get_min_terminal_cost())
                return Result(Status.SUCCESS, f"Goal reached! ({move_iter} iters)", execution_costs)

            while warmstart_count < hp['warmstart']:
                command = self.mppi.command(phy, self.goal.get_results, self.goal.cost, sub_time_s, num_samples)
                self.mppi_viz(self.goal, phy, command, sub_time_s)
                warmstart_count += 1
            command = self.mppi.command(phy, self.goal.get_results, self.goal.cost, sub_time_s, num_samples)
            self.mppi_viz(self.goal, phy, command, sub_time_s)

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

    def mppi_viz(self, goal, phy, command, sub_time_s):
        if not self.viz.p.mppi_rollouts:
            return

        sorted_traj_indices = np.argsort(self.mppi.cost)

        # viz
        i = None
        num_samples = self.mppi.cost.shape[0]
        for i in range(min(num_samples, 10)):
            sorted_traj_idx = sorted_traj_indices[i]
            cost_normalized = self.mppi.cost_normalized[sorted_traj_idx]
            c = cm.RdYlGn(1 - cost_normalized)
            result_i = tuple(r[sorted_traj_idx] for r in self.mppi.rollout_results)
            goal.viz_result(result_i, i, color=c, scale=0.002)
            rospy.sleep(0.01)

        cmd_rollout_results = rollout(phy.copy_data(), command[None], sub_time_s, get_result_func=goal.get_results)

        goal.viz_result(cmd_rollout_results, i, color='b', scale=0.004)

    def close(self):
        if self.mov is not None:
            self.mov.close()

    def sample_is_grasping(self, param):
        r = self.is_gasping_rng.uniform(size=self.n_g)
        is_grasping = (r < param).astype(int)
        return is_grasping
