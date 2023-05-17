import logging
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from enum import Enum, auto
from typing import Optional

import cma
import numpy as np
import rerun as rr
from colorama import Fore
from matplotlib import cm

import rospy
from mjregrasping.body_with_children import Objects
from mjregrasping.buffer import Buffer
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
    SHUTDOWN = auto()
    MOVE_TO_GOAL = auto()


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


def compute_grasp_error(phy):
    grasp_errors = []
    for i in range(phy.m.neq):
        eq = phy.m.eq(i)
        grasp_errors.append(np.sum(np.square(phy.d.body(eq.obj2id).xpos - phy.d.body(eq.obj1id).xpos)))
    return sum(grasp_errors) * 10


def vis_regrasp_solutions_and_costs(is_grasping, costs_lists, candidate_grasp_locations, all_queries, all_costs):
    # something representing the cost surface?
    cost_surface_map = {}
    n_g = len(all_queries[0])
    for query, cost in zip(all_queries, all_costs):
        x = query[1]
        y = query[0]
        z = cost / 1000
        if (x, y) not in cost_surface_map:
            cost_surface_map[x, y] = []
        cost_surface_map[x, y].append(z)

    # histograms
    width = 0.1
    cost_to_meters = 0.001
    depth = 0.05
    cost_names = ['f_is_same', 'f_new', 'f_new_mm', 'f_diff', 'f_diff_mm', 'f_goal']
    cost_colors = [
        [128, 128, 128],
        [255, 255, 0],
        [255, 0, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 0],
    ]

    def pos_transform(p):
        T = np.eye(4)
        T[0, 3] = p[0]
        T[1, 3] = p[1]
        T[2, 3] = p[2]
        return T

    for i, (costs_i, locations) in enumerate(zip(costs_lists, candidate_grasp_locations)):
        # TODO: draw one big outline box around the total cost for each solution
        z_offset = 0
        for cost_i, name, color_i in zip(costs_i, cost_names, cost_colors):
            z_i = np.clip(cost_i * cost_to_meters, 1e-3, 1e3)  # ensure non-zero
            box_entity_path = f'regrasp_costs/{i}/{name}'
            log_box(box_entity_path, np.array([width, depth, z_i]),
                    pos_transform([width * i, 0, z_i / 2 + z_offset]),
                    color=color_i)
            ext = {f'grasp {gripper_idx_to_eq_name(k)}': locations[k] for k in range(n_g)}
            ext['real cost'] = f'{cost_i:.2f}'
            ext['total cost'] = sum(costs_i)
            ext['is_grasping'] = ' '.join([str(g) for g in is_grasping])
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

        n_samples = self.p.n_samples
        horizon = self.p.horizon
        lambda_ = self.p.lambda_
        self.mppi = MujocoMPPI(pool=pool, nu=mppi_nu, seed=seed, num_samples=n_samples, noise_sigma=np.deg2rad(8),
                               horizon=horizon, lambda_=lambda_)
        self.dq_buffer = Buffer(12)
        self.max_dq = 0

        self.regrasp_idx = 0

    def run(self, phy):
        for self.iter in range(self.p.iters):
            if rospy.is_shutdown():
                self.close()
                return Result(Status.SHUTDOWN, "ROS shutdown", np.inf)

            logger.info(Fore.BLUE + f"Moving to goal" + Fore.RESET)
            move_result = self.move_to_goal(phy, self.p.max_move_to_goal_iters, is_planning=False)
            self.max_dq = 0  # reset "stuck" detection

            if move_result.status in [Status.SUCCESS, Status.FAILED, Status.SHUTDOWN]:
                self.close()
                return move_result

            grasp = self.compute_new_grasp(phy)
            logger.info(Fore.BLUE + f"Grasping {grasp}" + Fore.RESET)
            grasp_result = self.do_multi_gripper_regrasp(phy, grasp, max_iters=self.p.max_grasp_iters,
                                                         stop_if_failed=True, is_planning=False)
            self.max_dq = 0  # reset "stuck" detection
            if grasp_result.status in [Status.FAILED, Status.SHUTDOWN]:
                self.close()
                return grasp_result

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
        # # NOTE: just using for testing
        # if self.regrasp_idx % 2 == 0:
        #     return GraspState(self.rope_body_indices, np.array([0.483, 0.0]), np.array([1, 0]))
        # else:
        #     return GraspState(self.rope_body_indices, np.array([0.0, 0.95]), np.array([0, 1]))

        grasp0 = GraspState.from_mujoco(self.rope_body_indices, phy.m)
        all_binary_grasps = np.array([
            [1, 0],
            [0, 1],
            [1, 1],
            # [0, 0], # not very useful :)
        ])

        f_best = 1e9
        grasp_best = None
        for is_grasping in all_binary_grasps:
            logger.info(Fore.YELLOW + f'is_grasping={is_grasping}' + Fore.RESET)
            # Run CMA-ES to find the best grasp given the specific binary grasp state.
            # The objective function consists of solving a series of planning problems with MPPI.
            cma_idx = 0
            all_queries = []
            all_costs = []
            es = cma.CMAEvolutionStrategy(x0=grasp0.locations, sigma0=0.3, inopts={
                'popsize':   3,
                'bounds':    [0, 1],
                'seed':      1,
                'tolx':      1e-2,  # 1cm
                'maxfevals': 8,
                'tolfun':    10})  # how is this tolfun used?

            while not es.stop():
                if rospy.is_shutdown():
                    raise RuntimeError("ROS shutdown")

                candidate_grasp_locations = es.ask()  # from 0 to 1
                costs_lists = self.score_grasp_locations(candidate_grasp_locations, grasp0, is_grasping, phy)
                costs = [sum(costs_i) for costs_i in costs_lists]

                # Visualize!
                all_queries.extend(candidate_grasp_locations)
                all_costs.extend(costs)
                vis_regrasp_solutions_and_costs(is_grasping, costs_lists, candidate_grasp_locations, all_queries,
                                                all_costs)
                es.tell(candidate_grasp_locations, costs)
                print(es.result_pretty())

                cma_idx += 1

            if es.result.fbest < f_best:
                f_best = es.result.fbest
                grasp_best = GraspState(self.rope_body_indices, es.result.xbest, is_grasping)

        logger.info(Fore.GREEN + f"Best grasp: {grasp_best=}, {f_best=}" + Fore.RESET)
        return grasp_best

    def score_grasp_locations(self, candidate_grasp_locations, grasp0, is_grasping, phy):
        costs_lists = []
        for grasp_locations in candidate_grasp_locations:
            from time import perf_counter
            t0 = perf_counter()
            # copy model and data since each solution should be different/independent
            candidate_phy = copy(phy)

            grasp = GraspState(self.rope_body_indices, grasp_locations, is_grasping)

            f_is_same = 1000 if grasp0 == grasp else 0
            regrasp_result = self.do_multi_gripper_regrasp(candidate_phy, grasp, self.p.max_grasp_plan_iters,
                                                           is_planning=True)
            f_news, f_new_mms, f_diffs, f_diff_mms = regrasp_result.cost

            # Finally, MPPI to the goal
            move_result = self.move_to_goal(candidate_phy, self.p.max_plan_to_goal_iters, is_planning=True)
            f_goal = move_result.cost * self.p.f_goal_weight

            f_new = sum(f_news)
            f_new_mm = sum(f_new_mms)
            f_diff = sum(f_diffs) * self.p.f_diff_weight
            f_diff_mm = sum(f_diff_mms)
            costs_i = [f_is_same, f_new, f_new_mm, f_diff, f_diff_mm, f_goal]
            total_cost = sum(costs_i)
            logger.info(f'{grasp=} {total_cost=}')

            costs_lists.append(costs_i)
            # print(f"eval one solution's cost dt: {perf_counter() - t0:.1f}s")
        return costs_lists

    def do_multi_gripper_regrasp(self, phy, grasp, max_iters, is_planning: bool, stop_if_failed=False):
        settle_steps = self.p.plan_settle_steps if is_planning else self.p.settle_steps
        grasp0 = GraspState.from_mujoco(self.rope_body_indices, phy.m)
        is_new = grasp0.is_new(grasp)
        is_diff = grasp0.is_diff(grasp)
        needs_release = grasp0.needs_release(grasp)
        # For each gripper, if it needs a _new_ grasp, run MPPI to try to find a grasp and add the cost
        # NOTE: technically "order" of which gripper we consider first matters, we should in theory try all
        f_news = []
        f_new_mms = []
        f_diffs = []
        f_diff_mms = []
        for gripper_idx in range(self.n_g):
            if is_new[gripper_idx]:
                # plan the grasp
                f_new_result = self.do_single_gripper_grasp(phy, grasp, gripper_idx, max_iters, is_planning)
                if f_new_result.status in [Status.FAILED, Status.SHUTDOWN] and stop_if_failed:
                    return f_new_result
                f_news.append(f_new_result.cost)
                # activate new grasp
                change_eq(phy.m, gripper_idx_to_eq_name(gripper_idx), grasp.locations[gripper_idx],
                          self.rope_body_indices)
                settle(phy, self.p.grasp_sub_time_s, self.viz, is_planning, settle_steps)
                # add error for the grasp changing the state a lot, or for the eq constraint not being met
                f_new_mm_i = compute_grasp_error(phy)
                f_new_mms.append(f_new_mm_i)
        # deactivate
        for gripper_idx in range(self.n_g):
            if is_diff[gripper_idx] or needs_release[gripper_idx]:
                deactivate_eq(phy.m, gripper_idx_to_eq_name(gripper_idx))
                settle(phy, self.p.grasp_sub_time_s, self.viz, is_planning, settle_steps)
        # For each gripper, if it needs a different grasp, run MPPI to try to find a grasp and add the cost
        for gripper_idx in range(self.n_g):
            if is_diff[gripper_idx]:
                f_diff_result = self.do_single_gripper_grasp(phy, grasp, gripper_idx, max_iters, is_planning)
                if f_diff_result.status in [Status.FAILED, Status.SHUTDOWN] and stop_if_failed:
                    return f_diff_result
                f_diffs.append(f_diff_result.cost)
                change_eq(phy.m, gripper_idx_to_eq_name(gripper_idx), grasp.locations[gripper_idx],
                          self.rope_body_indices)
                settle(phy, self.p.grasp_sub_time_s, self.viz, is_planning, settle_steps)
                f_diff_mm_i = compute_grasp_error(phy)
                f_diff_mms.append(f_diff_mm_i)

        return Result(Status.SUCCESS, cost=(f_news, f_new_mms, f_diffs, f_diff_mms))

    def do_single_gripper_grasp(self, phy, grasp: GraspState, gripper_idx: int, max_iters: int, is_planning: bool):
        offset = grasp.offsets[gripper_idx]
        rope_body_to_grasp = phy.m.body(grasp.indices[gripper_idx])
        gripper_name = gripper_idx_to_eq_name(gripper_idx)

        self.mppi.reset()
        self.dq_buffer.reset()
        self.viz.viz(phy, is_planning)

        warmstart_count = 0
        cumulative_cost = 0
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
                return Result(Status.SUCCESS, f"Grasp successful", cumulative_cost)

            while warmstart_count < self.p.warmstart:
                command = self.mppi.command(phy, grasp_goal.get_results, grasp_goal.cost, self.p.grasp_sub_time_s)
                self.mppi_viz(grasp_goal, phy, command, self.p.grasp_sub_time_s)
                warmstart_count += 1
            command = self.mppi.command(phy, grasp_goal.get_results, grasp_goal.cost, self.p.grasp_sub_time_s)
            self.mppi_viz(grasp_goal, phy, command, self.p.grasp_sub_time_s)

            control_step(phy, command, sub_time_s=self.p.grasp_sub_time_s)
            self.viz.viz(phy, is_planning)
            if not is_planning:
                self.mov.render(phy.d)
            self.mppi.roll()

            if grasp_iter > max_iters:
                return Result(Status.FAILED, f"Failed to grasp after {grasp_iter} iters", cumulative_cost)

            grasp_iter += 1
            cumulative_cost += self.mppi.cost.min()

    def move_to_goal(self, phy, max_iters, is_planning: bool):
        self.mppi.reset()
        self.dq_buffer.reset()
        self.viz.viz(phy, is_planning)
        warmstart_count = 0
        cumulative_cost = 0
        move_iter = 0

        while True:
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown")

            self.goal.viz_goal(phy)
            if self.goal.satisfied(phy):
                return Result(Status.SUCCESS, "Goal reached!", cumulative_cost)

            while warmstart_count < self.p.warmstart:
                command = self.mppi.command(phy, self.goal.get_results, self.goal.cost, self.p.grasp_sub_time_s)
                self.mppi_viz(self.goal, phy, command, self.p.grasp_sub_time_s)
                warmstart_count += 1
            command = self.mppi.command(phy, self.goal.get_results, self.goal.cost, self.p.grasp_sub_time_s)
            self.mppi_viz(self.goal, phy, command, self.p.grasp_sub_time_s)

            control_step(phy, command, self.p.grasp_sub_time_s)
            self.viz.viz(phy, is_planning)
            if not is_planning:
                self.mov.render(phy.d)

            self.mppi.roll()

            needs_regrasp = self.check_needs_regrasp(phy.d)
            if needs_regrasp:
                return Result(Status.REGRASP, "Needs regrasp", cumulative_cost)

            if move_iter > max_iters:
                return Result(Status.FAILED, f"Gave up after {move_iter} iters", cumulative_cost)

            cumulative_cost += self.mppi.cost.min()
            move_iter += 1

    def mppi_viz(self, goal, phy, command, sub_time_s):
        if not self.p.mppi_rollouts:
            return

        sorted_traj_indices = np.argsort(self.mppi.cost)

        # viz
        i = None
        for i in range(min(self.mppi.num_samples, 10)):
            sorted_traj_idx = sorted_traj_indices[i]
            cost_normalized = self.mppi.cost_normalized[sorted_traj_idx]
            c = cm.RdYlGn(1 - cost_normalized)
            result_i = tuple(r[sorted_traj_idx] for r in self.mppi.rollout_results)
            goal.viz_result(result_i, i, color=c, scale=0.002)
            rospy.sleep(0.01)

        cmd_rollout_results = rollout(copy(phy), command[None], sub_time_s, get_result_func=goal.get_results)

        goal.viz_result(cmd_rollout_results, i, color='b', scale=0.004)

    def close(self):
        if self.mov is not None:
            self.mov.close()
