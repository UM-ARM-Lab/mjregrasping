import logging
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from enum import Enum, auto
from typing import Optional, List

import cma
import mujoco
import numpy as np
import rerun as rr
from colorama import Fore
from matplotlib import cm

import rospy
from mjregrasping.body_with_children import Objects
from mjregrasping.buffer import Buffer
from mjregrasping.goals import ObjectPointGoal, GraspBodyGoal
from mjregrasping.grasping import get_grasp_indices, get_grasp_constraints, deactivate_eq, change_eq
from mjregrasping.movie import MjMovieMaker
from mjregrasping.mujoco_mppi import MujocoMPPI
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

    def __init__(self, status: Status, msg: Optional[str] = None):
        self.status = status
        self.msg = msg

    def __str__(self):
        return f"{self.status.name} {self.msg}"

    def __repr__(self):
        return str(self)


def gripper_idx_to_eq_name(i):
    return 'left' if i == 0 else 'right'


def compute_grasp_error(d_before, d_after):
    state_change = np.sum(np.abs(d_before.qpos - d_after.qpos))
    # TODO: I don't really understand what qfrc_constraint is, so this might not be the right quantity
    constraint_force_change = np.sum(np.abs(d_after.qfrc_constraint - d_before.qfrc_constraint))
    grasp_error = constraint_force_change + state_change
    return grasp_error


def vis_regrasp_solutions_and_costs(cma_idx, costs, costs_lists, rope_indices, solutions, all_queries, all_costs):
    # Images representing the cost surface
    cost_surface_map = {}
    n_g = int(len(all_queries[0]) // 2)
    for query, cost in zip(all_queries, all_costs):
        y = (0 if query[0] == 0 else int(query[2]) - rope_indices.min())
        x = (0 if query[1] == 0 else int(query[3]) - rope_indices.min())
        z = cost / 1000
        if (x, y) not in cost_surface_map:
            cost_surface_map[x, y] = []
        cost_surface_map[x, y].append(z)
    cost_surface_mean_array = np.zeros([len(rope_indices) + 1, len(rope_indices) + 1])
    cost_surface_std_array = np.zeros([len(rope_indices) + 1, len(rope_indices) + 1])
    for (x, y), zs in cost_surface_map.items():
        cost_surface_mean_array[x, y] = np.mean(zs)
        cost_surface_std_array[x, y] = np.std(zs)
    rr.log_image('regrasp_cost_surface/mean', cost_surface_mean_array)
    rr.log_image('regrasp_cost_surface/std', cost_surface_std_array)

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

    for i, (costs_i, viz_solution) in enumerate(zip(costs_lists, solutions)):
        grasp_indices, is_grasping = remap_solution_to_params(n_g, viz_solution)
        z_offset = 0
        for cost_i, name, color_i in zip(costs_i, cost_names, cost_colors):
            z_i = np.clip(cost_i * cost_to_meters, 1e-3, 1e3)  # ensure non-zero
            box_entity_path = f'regrasp_costs/{cma_idx}/{i}/{name}'
            log_box(box_entity_path, np.array([width, depth, z_i]),
                    pos_transform([width * i, (2 * depth) * cma_idx, z_i / 2 + z_offset]), color=color_i)
            ext = {f'grasp {gripper_idx_to_eq_name(i)}': -1 if not is_grasping[i] else grasp_indices[i] for i in range(n_g)}
            ext['real cost'] = cost_i
            ext['total cost'] = sum(costs_i)
            rr.log_extension_components(box_entity_path, ext)
            z_offset += z_i


def remap_solution_to_params(n_g, solution):
    is_grasping = np.round(solution[:n_g]).astype(int) % 2
    grasp_indices = np.round(solution[n_g:]).astype(int)
    return grasp_indices, is_grasping


class RegraspMPC:

    def __init__(self, model, pool: ThreadPoolExecutor, viz: Viz, goal: ObjectPointGoal, objects: Objects,
                 seed: int = 1, mov: Optional[MjMovieMaker] = None):
        self.model = model
        # FIXME: since model may change over time, don't store it as a member variable
        self.viz = viz
        self.p = viz.p
        self.goal = goal
        self.objects = objects
        self.mov = mov

        n_samples = self.p.n_samples
        horizon = self.p.horizon
        lambda_ = self.p.lambda_
        self.mppi = MujocoMPPI(pool, self.model, seed=seed, num_samples=n_samples, noise_sigma=np.deg2rad(8),
                               horizon=horizon, lambda_=lambda_)
        self.buffer = Buffer(5)
        self.max_dq = 0
        self.regrasp_idx = 0

    def run(self, data):
        for i in range(self.p.iters):
            if rospy.is_shutdown():
                if self.mov is not None:
                    self.mov.close()
                return Result(Status.SHUTDOWN, "ROS shutdown")

            move_result = self.move_to_goal(data)

            if move_result.status in [Status.SUCCESS, Status.FAILED, Status.SHUTDOWN]:
                if self.mov is not None:
                    self.mov.close()
                return move_result

            regrasp_result = self.regrasp(data)
            if regrasp_result.status in [Status.SUCCESS, Status.FAILED, Status.SHUTDOWN]:
                if self.mov is not None:
                    self.mov.close()
                return regrasp_result

    def move_to_goal(self, data):
        """
        Runs MPPI with our novel cost function until the goal is reached or the robot is struck
        """
        sub_time_s = self.p.move_sub_time_s
        warmstart_count = 0
        self.mppi.reset()
        self.buffer.reset()
        self.viz.viz(self.model, data)
        move_iters = 0
        while True:
            if rospy.is_shutdown():
                return Result(Status.FAILED, "ROS shutdown")

            self.goal.viz_goal(data)
            if self.goal.satisfied(data):
                logger.info(Fore.GREEN + "Goal reached!" + Fore.RESET)
                return Result(Status.SUCCESS)

            if self.goal.last_any_points_useful != self.goal.any_points_useful:
                warmstart_count = 0
            while warmstart_count < self.p.warmstart:
                command = self.mppi.command(self.model, data, self.goal.get_results, self.goal.cost, sub_time_s)
                self.mppi_viz(self.goal, self.model, data, command, sub_time_s)
                warmstart_count += 1
            command = self.mppi.command(self.model, data, self.goal.get_results, self.goal.cost, sub_time_s)
            self.mppi_viz(self.goal, self.model, data, command, sub_time_s)

            control_step(self.model, data, command, sub_time_s)
            self.viz.viz(self.model, data)
            if self.mov is not None:
                self.mov.render(data)

            self.mppi.roll()

            needs_regrasp = self.check_needs_regrasp(data, command)
            if needs_regrasp:
                logger.warning("Needs regrasp!")
                return Result(Status.REGRASP)

            move_iters += 1
            if move_iters > self.p.max_move_to_goal_iters:
                return Result(Status.FAILED, f"Failed to reach goal or detect regrasp after {move_iters} iters")

    def check_needs_regrasp(self, data, command):
        self.buffer.insert(data.qpos[self.objects.val.qpos_indices])
        qposs = np.array(self.buffer.data)
        if len(qposs) < 2:
            dq = 0
        else:
            dq = np.abs(qposs[1:] - qposs[:-1]).mean()
        self.max_dq = max(self.max_dq, dq)
        has_not_moved = dq < self.p.frac_max_dq * self.max_dq
        needs_regrasp = self.buffer.full() and has_not_moved
        rr.log_scalar('needs_regrasp/dq', dq)
        rr.log_scalar('needs_regrasp/max_dq', self.max_dq)
        return needs_regrasp

    def regrasp(self, data):
        sub_time_s = self.p.grasp_sub_time_s
        current_grasp = get_grasp_indices(self.model)
        new_grasp = self.compute_new_grasp(data)
        self.regrasp_idx += 1

        if np.all(new_grasp == current_grasp):
            logger.info(Fore.GREEN + "No need to regrasp!" + Fore.RESET)
            return
        elif np.all(new_grasp != current_grasp) and np.all(new_grasp != -1):
            raise NotImplementedError("Cannot regrasp with both grippers at once")
        elif new_grasp[0] != -1:
            gripper_idx = 0
        else:
            gripper_idx = 1

        rope_body_to_grasp = self.model.body(new_grasp[gripper_idx])
        # NOTE: why does body.pos give the wrong value?
        logger.info(f"grasping {rope_body_to_grasp.name}")

        warmstart_count = 0
        self.mppi.reset()
        self.buffer.reset()
        self.viz.viz(self.model, data)
        regrasp_iters = 0
        while True:
            if rospy.is_shutdown():
                return Result(Status.SHUTDOWN, "ROS shutdown")

            # recreate the goal at each time step so the goal point is updated
            regrasp_goal = GraspBodyGoal(model=self.model,
                                         body_id_to_grasp=rope_body_to_grasp.id,
                                         goal_radius=0.015,
                                         gripper_idx=gripper_idx,
                                         viz=self.viz,
                                         objects=self.objects)

            regrasp_goal.viz_goal(data)
            if regrasp_goal.satisfied(data):
                logger.info(Fore.GREEN + "Regrasp successful!" + Fore.RESET)
                break

            while warmstart_count < self.p.warmstart:
                command = self.mppi.command(self.model, data, regrasp_goal.get_results, regrasp_goal.cost, sub_time_s)
                self.mppi_viz(regrasp_goal, self.model, data, command, sub_time_s)
                warmstart_count += 1
            command = self.mppi.command(self.model, data, regrasp_goal.get_results, regrasp_goal.cost, sub_time_s)
            self.mppi_viz(regrasp_goal, self.model, data, command, sub_time_s)

            # NOTE: what if sub_time_s was proportional to the distance to goal? or the cost?
            control_step(self.model, data, command, sub_time_s=self.p.grasp_sub_time_s)
            self.viz.viz(self.model, data)
            if self.mov is not None:
                self.mov.render(data)

            self.mppi.roll()

            regrasp_iters += 1
            if regrasp_iters > self.p.max_regrasp_iters:
                return Result(Status.FAILED, f"Failed to regrasp after {regrasp_iters} iters")

        # change the mujoco grasp constraints
        for eq, grasp_idx in zip(get_grasp_constraints(self.model), new_grasp):
            if grasp_idx == -1:
                eq.active[:] = 0
            else:
                eq.obj2id = grasp_idx
                eq.active[:] = 1
                eq.data[3:6] = 0

        return Result(Status.MOVE_TO_GOAL)

    def mppi_viz(self, goal, m, d, command, sub_time_s):
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

        cmd_rollout_results = rollout(m, copy(d), command[None], sub_time_s,
                                      get_result_func=goal.get_results)

        goal.viz_result(cmd_rollout_results, i, color='b', scale=0.004)

    def compute_new_grasp(self, data):
        # run CMA-ES to find the best grasp
        # where the objective function comes from solving a series of planning problems with MPPI
        current_grasp = get_grasp_indices(self.model)
        n_g = len(current_grasp)
        is_grasping0 = (current_grasp != -1)
        rope_indices = np.array(self.objects.rope.body_indices)
        # replace -1 with rope_indices.min() since -1 is not a valid index,
        # and we're representing grasping with a separate binary variable
        grasp_indices0 = np.maximum(current_grasp, rope_indices.min())
        x0 = np.concatenate([is_grasping0, grasp_indices0])
        # in order to sample a binary variable, we set the bounds to be from 0 to 10 and then round and mod by 2
        lower_bounds = np.concatenate([np.zeros(n_g), np.full(n_g, rope_indices.min())])
        upper_bounds = np.concatenate([np.ones(n_g) * 10, np.full(n_g, rope_indices.max())])
        sigma0 = np.concatenate([np.ones(n_g), np.full(n_g, 5)])
        es = cma.CMAEvolutionStrategy(x0=x0, sigma0=3.33, inopts={
            'CMA_stds':          sigma0,
            'popsize':           10,
            'integer_variables': range(2 * n_g),
            'bounds':            [lower_bounds, upper_bounds],
            'seed':              1})

        cma_idx = 0
        all_queries = []
        all_costs = []
        while not es.stop():
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown")

            solutions = es.ask()
            # costs = self.compute_grasp_costs(solutions, is_grasping0, grasp_indices0, data0)
            costs_lists = []
            costs = []
            for solution in solutions:
                from time import perf_counter
                t0 = perf_counter()
                # copy since each solution should be different/independent
                d = copy(data)
                m = copy(self.model)
                grasp_indices, is_grasping = remap_solution_to_params(n_g, solution)
                print(is_grasping, grasp_indices)

                same = np.logical_or(np.logical_and(np.logical_not(is_grasping0), np.logical_not(is_grasping)),
                                     np.logical_and(is_grasping0,
                                                    np.logical_and(is_grasping, (grasp_indices0 == grasp_indices))))
                f_is_same = 1000 if np.all(same) else 0
                f_news = []
                f_new_mms = []
                f_diffs = []
                f_diff_mms = []
                # For each gripper, if it needs a _new_ grasp, run MPPI to try to find a grasp and add the cost
                for i in range(n_g):
                    new_grasp = not is_grasping0[i] and is_grasping[i]
                    if new_grasp:
                        # plan the grasp
                        f_news.append(self.plan_grasp(m, d, is_grasping, grasp_indices, i))
                        # activate new grasp
                        change_eq(m, gripper_idx_to_eq_name(i), grasp_indices[i])
                        d_before = copy(d)
                        settle(d, m, self.p.grasp_sub_time_s, self.viz)
                        # add error for the grasp changing the state a lot, or for the eq cosntraint not being met
                        f_new_mms.append(compute_grasp_error(d_before, d))
                # deactivate
                for i in range(n_g):
                    diff_grasp = is_grasping0[i] and is_grasping[i] and grasp_indices0[i] != grasp_indices[i]
                    release = is_grasping0[i] and not is_grasping[i]
                    if release or diff_grasp:
                        deactivate_eq(m, gripper_idx_to_eq_name(i))
                        settle(d, m, self.p.grasp_sub_time_s, self.viz)
                # For each gripper, if it needs a _different_ grasp, run MPPI to try to find a grasp and add the cost
                for i in range(n_g):
                    diff_grasp = is_grasping0[i] and is_grasping[i] and grasp_indices0[i] != grasp_indices[i]
                    if diff_grasp:
                        f_diffs.append(self.plan_grasp(m, d, is_grasping, grasp_indices, i))
                        change_eq(m, gripper_idx_to_eq_name(i), grasp_indices[i])
                        d_before = copy(d)
                        settle(d, m, self.p.grasp_sub_time_s, self.viz)
                        f_diff_mms.append(compute_grasp_error(d_before, d))
                # Finally, MPPI to the goal
                f_goal = self.plan_to_goal(m, d) * 0.01

                f_new = sum(f_news)
                f_new_mm = sum(f_new_mms)
                f_diff = sum(f_diffs)
                f_diff_mm = sum(f_diff_mms)
                costs_i = [f_is_same, f_new, f_new_mm, f_diff, f_diff_mm, f_goal]
                total_cost = sum(costs_i)
                print(total_cost)

                costs.append(total_cost)
                costs_lists.append(costs_i)
                print(f"eval one solution's cost dt: {perf_counter() - t0:.4f}")

            # Visualize!
            all_queries.extend(solutions)
            all_costs.extend(costs)
            vis_regrasp_solutions_and_costs(cma_idx, costs, costs_lists, rope_indices, solutions, all_queries, all_costs)
            es.tell(solutions, costs)
            print(es.result_pretty())

            cma_idx += 1

        return es.result.xbest
        #####################################
        # NOTE: just using for testing
        # if self.regrasp_idx % 2 == 0:
        #     return np.array([57, -1])
        # else:
        #     return np.array([-1, 61])

    def compute_grasp_costs(self, solutions: List[np.ndarray], is_grasping0, grasp_indices0, data0: mujoco.MjData):
        pass

    def plan_grasp(self, m, d, is_grasping, grasp_indices, i):
        rope_body_to_grasp = m.body(grasp_indices[i])
        logger.info(f"considering grasping {rope_body_to_grasp.name} with {gripper_idx_to_eq_name(i)} gripper")

        self.mppi.reset()
        self.buffer.reset()
        self.viz.viz(m, d)

        warmstart_count = 0
        cumulative_cost = 0
        regrasp_goal = GraspBodyGoal(model=m,
                                     body_id_to_grasp=rope_body_to_grasp.id,
                                     goal_radius=0.015,
                                     gripper_idx=i,
                                     viz=self.viz,
                                     objects=self.objects)
        for grasp_iter in range(self.p.max_regrasp_iters):
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown")

            regrasp_goal.viz_goal(d)
            if regrasp_goal.satisfied(d):
                logger.info(Fore.GREEN + "Regrasp successful!" + Fore.RESET)
                break

            while warmstart_count < self.p.warmstart:
                command = self.mppi.command(m, d, regrasp_goal.get_results, regrasp_goal.cost, self.p.grasp_sub_time_s)
                self.mppi_viz(regrasp_goal, m, d, command, self.p.grasp_sub_time_s)
                warmstart_count += 1
            command = self.mppi.command(m, d, regrasp_goal.get_results, regrasp_goal.cost, self.p.grasp_sub_time_s)
            self.mppi_viz(regrasp_goal, m, d, command, self.p.grasp_sub_time_s)

            control_step(m, d, command, sub_time_s=self.p.grasp_sub_time_s)
            self.viz.viz(m, d)
            self.mppi.roll()

            cumulative_cost += self.mppi.cost.min()

        return cumulative_cost

    def plan_to_goal(self, m, d):
        self.mppi.reset()
        self.buffer.reset()
        self.viz.viz(m, d)
        warmstart_count = 0
        cumulative_cost = 0

        for grasp_iter in range(self.p.max_move_to_goal_iters):
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown")

            self.goal.viz_goal(d)
            if self.goal.satisfied(d):
                logger.info(Fore.GREEN + "Goal reached!" + Fore.RESET)
                break

            while warmstart_count < self.p.warmstart:
                command = self.mppi.command(m, d, self.goal.get_results, self.goal.cost, self.p.grasp_sub_time_s)
                self.mppi_viz(self.goal, m, d, command, self.p.grasp_sub_time_s)
                warmstart_count += 1
            command = self.mppi.command(m, d, self.goal.get_results, self.goal.cost, self.p.grasp_sub_time_s)
            self.mppi_viz(self.goal, m, d, command, self.p.grasp_sub_time_s)

            control_step(m, d, command, sub_time_s=self.p.grasp_sub_time_s)
            self.viz.viz(m, d)
            self.mppi.roll()

            cumulative_cost += self.mppi.cost.min()

        return cumulative_cost
