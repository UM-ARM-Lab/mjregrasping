import logging
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from enum import Enum, auto
from typing import Optional

import numpy as np
import rerun as rr
from colorama import Fore
from matplotlib import cm

import rospy
from mjregrasping.body_with_children import Objects
from mjregrasping.buffer import Buffer
from mjregrasping.goals import ObjectPointGoal, GraspBodyGoal
from mjregrasping.grasping import get_grasp_indices, get_grasp_constraints
from mjregrasping.movie import MjMovieMaker
from mjregrasping.mujoco_mppi import MujocoMPPI
from mjregrasping.rollout import control_step, rollout
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


class RegraspMPC:

    def __init__(self, model, pool: ThreadPoolExecutor, viz: Viz, goal: ObjectPointGoal, objects: Objects,
                 seed: int = 1, mov: Optional[MjMovieMaker] = None):
        self.model = model
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
                command = self.mppi.command(data, self.goal.get_results, self.goal.cost, sub_time_s)
                self.mppi_viz(self.goal, data, command, sub_time_s)
                warmstart_count += 1
            command = self.mppi.command(data, self.goal.get_results, self.goal.cost, sub_time_s)
            self.mppi_viz(self.goal, data, command, sub_time_s)

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
        zero_command = np.linalg.norm(command) < self.p.min_command
        needs_regrasp = self.buffer.full() and (has_not_moved or zero_command)
        rr.log_scalar('needs_regrasp/dq', dq)
        rr.log_scalar('needs_regrasp/max_dq', self.max_dq)
        rr.log_scalar('needs_regrasp/command_norm', np.linalg.norm(command))
        return needs_regrasp

    def regrasp(self, data):
        sub_time_s = self.p.grasp_sub_time_s
        # new_grasp = self.compute_new_grasp(data)
        current_grasp = get_grasp_indices(self.model)
        # FIXME: this is a hack
        if self.regrasp_idx % 2 == 0:
            new_grasp = np.array([57, -1])
        else:
            new_grasp = np.array([-1, 60])
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
                command = self.mppi.command(data, regrasp_goal.get_results, regrasp_goal.cost, sub_time_s)
                self.mppi_viz(regrasp_goal, data, command, sub_time_s)
                warmstart_count += 1
            command = self.mppi.command(data, regrasp_goal.get_results, regrasp_goal.cost, sub_time_s)
            self.mppi_viz(regrasp_goal, data, command, sub_time_s)

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

    def compute_new_grasp(self, data):
        pass

    def mppi_viz(self, goal, data, command, sub_time_s):
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

        cmd_rollout_results = rollout(self.model, copy(data), command[None], sub_time_s,
                                      get_result_func=goal.get_results)

        goal.viz_result(cmd_rollout_results, i, color='b', scale=0.004)
