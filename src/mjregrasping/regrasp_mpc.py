import logging
from time import perf_counter
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
from time import perf_counter
from mjregrasping.change_grasp_eq import change_eq
from mjregrasping.goals import ObjectPointGoal, GraspRopeGoal
from mjregrasping.grasp_state import GraspState
from mjregrasping.grasping import deactivate_eq
from mjregrasping.movie import MjMovieMaker
from mjregrasping.mujoco_mppi import MujocoMPPI
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


def compute_eq_error(phy: Physics):
    eq_errs = []
    for i in range(phy.m.neq):
        eq = phy.m.eq(i)
        if eq.active:
            b2 = phy.d.body(eq.obj2id)
            b1 = phy.d.body(eq.obj1id)
            b1_offset = eq.data[0:3]
            b2_offset = eq.data[3:6]
            b1_offset_in_world = np.zeros(3)
            mujoco.mju_trnVecPose(b1_offset_in_world, b1.xpos, b1.xquat, b1_offset)
            b2_offset_in_world = np.zeros(3)
            mujoco.mju_trnVecPose(b2_offset_in_world, b2.xpos, b2.xquat, b2_offset)
            eq_err = np.linalg.norm(b1_offset_in_world - b2_offset_in_world)
            eq_errs.append(eq_err)
    return sum(eq_errs)


def vis_regrasp_solutions_and_costs(costs_lists: List[Dict], candidate_grasps: List[GraspState]):
    # histograms
    n_g = 2
    width = 0.1
    cost_to_meters = 0.5
    depth = 0.05
    cost_colors = {
        'f_is_same':     [128, 0, 128],
        'f_all_0':       [128, 0, 128],
        'f_goal':        [0, 255, 0],
        'f_new':         [0, 0, 255],
        'f_diff':        [0, 0, 255],
        'f_new_eq_err':  [255, 0, 0],
        'f_diff_eq_err': [255, 0, 0],
    }

    def pos_transform(p):
        T = np.eye(4)
        T[0, 3] = p[0]
        T[1, 3] = p[1]
        T[2, 3] = p[2]
        return T

    for i, (costs_i, grasp) in enumerate(zip(costs_lists, candidate_grasps)):
        # TODO: draw one big outline box around the total cost for each solution
        z_offset = 0
        for name, cost_i in costs_i.items():
            color_i = cost_colors[name]
            z_i = np.clip(cost_i * cost_to_meters, 1e-3, 1e3)  # ensure non-zero
            box_entity_path = f'regrasp_costs/{i}/{name}'
            log_box(box_entity_path, np.array([width, depth, z_i]),
                    pos_transform([width * i, 0, z_i / 2 + z_offset]),
                    color=color_i)
            ext = {f'grasp {gripper_idx_to_eq_name(k)}': grasp.locations[k] for k in range(n_g)}
            ext[name] = f'{cost_i:.3f}'
            ext['total cost'] = sum(costs_i.values())
            ext['is_grasping'] = ' '.join([str(g) for g in grasp.is_grasping])
            rr.log_extension_components(box_entity_path, ext)
            z_offset += z_i


class RegraspMPC:

    def __init__(self, mppi_nu: int, pool: ThreadPoolExecutor, viz: Viz, goal: ObjectPointGoal, objects: Objects,
                 seed: int = 1, mov: Optional[MjMovieMaker] = None):
        self.viz = viz
        self.p = viz.p
        self.goal = goal
        self.objects = objects
        self.rope_body_indices = np.array(self.objects.rope.body_indices)
        self.n_g = 2
        self.mov = mov

        horizon = self.p.horizon
        lambda_ = self.p.lambda_
        self.mppi = MujocoMPPI(pool=pool, nu=mppi_nu, seed=seed, noise_sigma=np.deg2rad(4), horizon=horizon,
                               lambda_=lambda_)
        self.dq_buffer = Buffer(12)
        self.max_dq = 0

        self.regrasp_idx = 0

    def run(self, phy):
        for self.iter in range(self.p.iters):
            if rospy.is_shutdown():
                self.close()
                raise RuntimeError("ROS shutdown")

            logger.info(Fore.BLUE + f"Moving to goal" + Fore.RESET)
            t0 = perf_counter()
            move_result = self.move_to_goal(phy, self.p.max_move_to_goal_iters, is_planning=False,
                                            sub_time_s=self.p.move_sub_time_s)
            print(f"move: {perf_counter() - t0:.2f}")
            self.max_dq = 0  # reset "stuck" detection

            if move_result.status in [Status.SUCCESS]:
                self.close()
                return move_result

            while True:
                # NOTE: currently testing what happens if we let it regrasp if a grasp fails
                logger.info(Fore.BLUE + f"Computing New Grasp" + Fore.RESET)
                grasp = self.compute_new_grasp(phy)
                logger.info(Fore.BLUE + f"Grasping {grasp}" + Fore.RESET)
                grasp_result = self.do_multi_gripper_regrasp(phy, grasp, max_iters=self.p.max_grasp_iters,
                                                             is_planning=False, sub_time_s=self.p.grasp_sub_time_s,
                                                             stop_if_failed=True)
                self.max_dq = 0  # reset "stuck" detection
                if grasp_result.status == Status.SUCCESS:
                    break

            self.regrasp_idx += 1

    def check_needs_regrasp(self, data):
        self.dq_buffer.insert(data.qpos[self.objects.val.qpos_indices])
        qposs = np.array(self.dq_buffer.data)
        if len(qposs) < 2:
            dq = 0
        else:
            # distance between the newest and oldest q in the buffer
            # the mean takes the average across joints.
            dq = (np.abs(qposs[-1] - qposs[0]) / len(self.dq_buffer)).mean()
        self.max_dq = max(self.max_dq, dq)
        has_not_moved = dq < self.p.frac_max_dq * self.max_dq
        needs_regrasp = self.dq_buffer.full() and has_not_moved
        rr.log_scalar('needs_regrasp/dq', dq)
        rr.log_scalar('needs_regrasp/max_dq', self.max_dq)
        return needs_regrasp

    def compute_new_grasp(self, phy: Physics):
        #####################################
        # # # NOTE: just using for testing
        # if self.regrasp_idx == 0:
        #     return GraspState(self.rope_body_indices, np.array([0.483, 0.0]), np.array([1, 0]))

        grasp0 = GraspState.from_mujoco(self.rope_body_indices, phy.m)
        all_binary_grasps = np.array([
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0],  # not very useful :)
        ])

        f_best = 1e9
        grasp_best = None
        for is_grasping in all_binary_grasps:
            logger.info(Fore.YELLOW + f'is_grasping={is_grasping}' + Fore.RESET)
            # Run CMA-ES to find the best grasp given the specific binary grasp state.
            # The objective function consists of solving a series of planning problems with MPPI.
            cma_idx = 0
            es = cma.CMAEvolutionStrategy(x0=grasp0.locations, sigma0=0.3, inopts={
                'popsize':   3,
                'bounds':    [0, 1],
                'seed':      1,
                'tolx':      1e-2,  # 1cm
                'maxfevals': 8,
                'tolfun':    0.05})  # how is this tolfun used?

            while not es.stop():
                if rospy.is_shutdown():
                    raise RuntimeError("ROS shutdown")

                candidate_grasp_locations = es.ask()  # from 0 to 1
                costs_lists = []
                candidate_grasps = []
                for grasp_locations in candidate_grasp_locations:
                    grasp = GraspState(self.rope_body_indices, grasp_locations, is_grasping)
                    candidate_grasps.append(grasp)
                    costs_i = self.score_grasp_location(phy, grasp0, grasp)
                    total_cost = sum(costs_i.values())
                    logger.info(f'{grasp=} {total_cost=}')
                    costs_lists.append(costs_i)
                costs = [sum(costs_i.values()) for costs_i in costs_lists]

                # Visualize!
                vis_regrasp_solutions_and_costs(costs_lists, candidate_grasps)
                es.tell(candidate_grasp_locations, costs)
                print(es.result_pretty())

                cma_idx += 1

            if es.result.fbest < f_best:
                f_best = es.result.fbest
                grasp_best = GraspState(self.rope_body_indices, es.result.xbest, is_grasping)

        logger.info(Fore.GREEN + f"Best grasp: {grasp_best=}, {f_best=:.2f}" + Fore.RESET)
        return grasp_best

    def score_grasp_location(self, phy: Physics, grasp0: GraspState, grasp: GraspState):
        # copy model and data since each solution should be different/independent
        candidate_phy = phy.copy_all()

        if np.all(np.logical_not(grasp.is_grasping)):
            return {'f_all_0': 1000}

        if grasp0 == grasp:
            return {'f_is_same': 1000}

        regrasp_result = self.do_multi_gripper_regrasp(candidate_phy, grasp, self.p.max_grasp_plan_iters,
                                                       is_planning=True, sub_time_s=self.p.plan_sub_time_s)
        f_news, f_new_eq_errs, f_diffs, f_diff_eq_errs = regrasp_result.cost

        # Finally, MPPI to the goal
        t0 = perf_counter()
        move_result = self.move_to_goal(candidate_phy, self.p.max_plan_to_goal_iters, is_planning=True,
                                        sub_time_s=self.p.plan_sub_time_s)
        print(f"move: {perf_counter() - t0:.2f}")

        f_goal = move_result.cost * self.p.f_goal_weight
        f_new = sum(f_news) * self.p.f_grasp_weight
        f_new_eq_err = sum(f_new_eq_errs) * self.p.f_eq_err_weight
        f_diff = sum(f_diffs) * self.p.f_grasp_weight
        f_diff_eq_err = sum(f_diff_eq_errs) * self.p.f_eq_err_weight

        return {
            'f_new':         f_new,
            'f_new_eq_err':  f_new_eq_err,
            'f_diff':        f_diff,
            'f_diff_eq_err': f_diff_eq_err,
            'f_goal':        f_goal,
        }

    def do_multi_gripper_regrasp(self, phy, grasp, max_iters, is_planning: bool, sub_time_s: float,
                                 stop_if_failed=False):
        settle_steps = self.p.plan_settle_steps if is_planning else self.p.settle_steps
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
                t0 = perf_counter()
                f_new_result = self.do_single_gripper_grasp(phy, grasp, gripper_idx, max_iters, is_planning, sub_time_s)
                print(f"grasp: {perf_counter() - t0:.2f}")
                if f_new_result.status in [Status.FAILED] and stop_if_failed:
                    return f_new_result
                f_news.append(f_new_result.cost)
                # activate new grasp
                change_eq(phy.m, gripper_idx_to_eq_name(gripper_idx), grasp.locations[gripper_idx],
                          self.rope_body_indices)
                settle(phy, sub_time_s, self.viz, is_planning, settle_steps)
                # add error for the grasp changing the state a lot, or for the eq constraint not being met
                # mm means 'mismatch'
                f_new_eq_err_i = compute_eq_error(phy)
                f_new_eq_errs.append(f_new_eq_err_i)
        # deactivate
        for gripper_idx in range(self.n_g):
            if is_diff[gripper_idx] or needs_release[gripper_idx]:
                deactivate_eq(phy.m, gripper_idx_to_eq_name(gripper_idx))
                settle(phy, sub_time_s, self.viz, is_planning, settle_steps)
        # For each gripper, if it needs a different grasp, run MPPI to try to find a grasp and add the cost
        for gripper_idx in range(self.n_g):
            if is_diff[gripper_idx]:
                t0 = perf_counter()
                f_diff_result = self.do_single_gripper_grasp(phy, grasp, gripper_idx, max_iters, is_planning,
                                                             sub_time_s)
                print(f"grasp: {perf_counter() - t0:.2f}")
                if f_diff_result.status in [Status.FAILED] and stop_if_failed:
                    return f_diff_result
                f_diffs.append(f_diff_result.cost)
                change_eq(phy.m, gripper_idx_to_eq_name(gripper_idx), grasp.locations[gripper_idx],
                          self.rope_body_indices)
                settle(phy, sub_time_s, self.viz, is_planning, settle_steps)
                f_diff_eq_err_i = compute_eq_error(phy)
                f_diff_eq_errs.append(f_diff_eq_err_i)

        return Result(Status.SUCCESS, cost=(f_news, f_new_eq_errs, f_diffs, f_diff_eq_errs))

    def do_single_gripper_grasp(self, phy, grasp: GraspState, gripper_idx: int, max_iters: int, is_planning: bool,
                                sub_time_s: float):
        offset = grasp.offsets[gripper_idx]
        rope_body_to_grasp = phy.m.body(grasp.indices[gripper_idx])

        self.mppi.reset()
        self.dq_buffer.reset()
        self.viz.viz(phy, is_planning)

        warmstart_count = 0
        execution_cost = 0
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
                execution_cost += self.mppi.get_min_terminal_cost()
                return Result(Status.SUCCESS, f"Grasp successful", execution_cost)

            while warmstart_count < self.p.warmstart:
                command = self.mppi.command(phy, grasp_goal.get_results, grasp_goal.cost, sub_time_s,
                                            self.p.num_samples)
                self.mppi_viz(grasp_goal, phy, command, sub_time_s)
                warmstart_count += 1
            command = self.mppi.command(phy, grasp_goal.get_results, grasp_goal.cost, sub_time_s, self.p.num_samples)
            self.mppi_viz(grasp_goal, phy, command, sub_time_s)

            control_step(phy, command, sub_time_s)
            self.viz.viz(phy, is_planning)
            if not is_planning and self.mov:
                self.mov.render(phy.d)
            self.mppi.roll()

            if grasp_iter > max_iters:
                execution_cost += self.mppi.get_min_terminal_cost()
                return Result(Status.FAILED, f"Failed to grasp after {grasp_iter} iters", execution_cost)

            grasp_iter += 1
            execution_cost += self.mppi.get_first_step_cost() * self.p.running_cost_weight

    def move_to_goal(self, phy, max_iters, is_planning: bool, sub_time_s: float):
        self.mppi.reset()
        self.dq_buffer.reset()
        self.viz.viz(phy, is_planning)
        warmstart_count = 0
        execution_cost = 0
        move_iter = 0

        while True:
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown")

            self.goal.viz_goal(phy)
            if self.goal.satisfied(phy):
                execution_cost += self.mppi.get_min_terminal_cost()
                return Result(Status.SUCCESS, "Goal reached!", execution_cost)

            while warmstart_count < self.p.warmstart:
                command = self.mppi.command(phy, self.goal.get_results, self.goal.cost, sub_time_s, self.p.num_samples)
                self.mppi_viz(self.goal, phy, command, sub_time_s)
                warmstart_count += 1
            command = self.mppi.command(phy, self.goal.get_results, self.goal.cost, sub_time_s, self.p.num_samples)
            self.mppi_viz(self.goal, phy, command, sub_time_s)

            control_step(phy, command, sub_time_s)
            self.viz.viz(phy, is_planning)
            if not is_planning and self.mov:
                self.mov.render(phy.d)

            self.mppi.roll()

            needs_regrasp = self.check_needs_regrasp(phy.d)
            if needs_regrasp:
                execution_cost += self.mppi.get_min_terminal_cost()
                return Result(Status.REGRASP, "Needs regrasp", execution_cost)

            if move_iter > max_iters:
                execution_cost += self.mppi.get_min_terminal_cost()
                return Result(Status.FAILED, f"Gave up after {move_iter} iters", execution_cost)

            execution_cost += self.mppi.get_first_step_cost() * self.p.running_cost_weight
            move_iter += 1

    def mppi_viz(self, goal, phy, command, sub_time_s):
        if not self.p.mppi_rollouts:
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
