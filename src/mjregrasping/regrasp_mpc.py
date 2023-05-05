import logging
from concurrent.futures import ThreadPoolExecutor
from copy import copy

import numpy as np
import rerun as rr
from colorama import Fore
from matplotlib import cm

import rospy
from mjregrasping.body_with_children import BodyWithChildren
from mjregrasping.buffer import Buffer
from mjregrasping.goals import ObjectPointGoal, GraspBodyGoal
from mjregrasping.grasping import get_grasp_indices, get_grasp_constraints
from mjregrasping.mujoco_mppi import MujocoMPPI
from mjregrasping.mujoco_visualizer import plot_lines_rviz
from mjregrasping.rollout import control_step, rollout
from mjregrasping.viz import Viz

logger = logging.getLogger(f'rosout.{__name__}')


class RegraspMPC:

    def __init__(self, model, pool: ThreadPoolExecutor, viz: Viz):
        self.model = model
        self.viz = viz
        self.p = viz.p
        self.val = BodyWithChildren(model, 'val_base')
        self.rope = BodyWithChildren(model, 'rope')
        # TODO: move this outside
        self.obstacle = BodyWithChildren(model, 'computer_rack')
        self.goal = ObjectPointGoal(model=self.model,
                                    # TODO: take in the whole viz object instead of viz_pubs and p separately
                                    viz_pubs=viz.pubs,
                                    # TODO: move this outside
                                    goal_point=np.array([0.78, 0.04, 1.27]),
                                    body_idx=-1,
                                    goal_radius=0.05,
                                    val=self.val,
                                    rope=self.rope,
                                    obstacle=self.obstacle,
                                    p=viz.p)

        n_samples = self.p.n_samples
        horizon = self.p.horizon
        lambda_ = self.p.lambda_
        self.mppi = MujocoMPPI(pool, self.model, num_samples=n_samples, noise_sigma=np.deg2rad(8), horizon=horizon,
                               lambda_=lambda_)
        self.buffer = Buffer(5)
        self.max_dq = 0
        self.regrasp_idx = 0

    def run(self, data):
        for i in range(self.p.iters):
            if rospy.is_shutdown():
                break

            done = self.move_to_goal(data)
            if done:
                break

            self.regrasp(data)

    def move_to_goal(self, data):
        """
        Runs MPPI with our novel cost function until the goal is reached or the robot is struck
        """
        sub_time_s = self.p.move_sub_time_s
        warmstart_count = 0
        self.mppi.reset()
        self.buffer.reset()
        self.viz.viz(self.model, data)
        while True:
            if rospy.is_shutdown():
                break

            self.goal.viz(data)
            if self.goal.satisfied(data):
                logger.info(Fore.GREEN + "Goal reached!" + Fore.RESET)
                return True

            while warmstart_count < self.p.warmstart:
                command = self.mppi.command(data, self.goal.get_results, self.goal.cost, sub_time_s)
                self.mppi_viz(self.goal, data, command, sub_time_s)
                warmstart_count += 1
            command = self.mppi.command(data, self.goal.get_results, self.goal.cost, sub_time_s)
            self.mppi_viz(self.goal, data, command, sub_time_s)

            control_step(self.model, data, command, sub_time_s)
            self.viz.viz(self.model, data)

            self.mppi.roll()

            needs_regrasp = self.check_needs_regrasp(data, command)
            if needs_regrasp:
                logger.warning("Needs regrasp!")
                break

        return False

    def check_needs_regrasp(self, data, command):
        self.buffer.insert(data.qpos[self.val.qpos_indices])
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
        while True:
            if rospy.is_shutdown():
                break

            # recreate the goal at each time step so the goal point is updated
            regrasp_goal = GraspBodyGoal(model=self.model,
                                         body_id_to_grasp=rope_body_to_grasp.id,
                                         goal_radius=0.015,
                                         gripper_idx=gripper_idx,
                                         visualizer=self.viz.pubs,
                                         val=self.val,
                                         rope=self.rope,
                                         obstacle=self.obstacle,
                                         p=self.p)

            regrasp_goal.viz(data)
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

            self.mppi.roll()

        # change the mujoco grasp constraints
        for eq, grasp_idx in zip(get_grasp_constraints(self.model), new_grasp):
            if grasp_idx == -1:
                eq.active[:] = 0
            else:
                eq.obj2id = grasp_idx
                eq.active[:] = 1
                eq.data[3:6] = 0

    def compute_new_grasp(self, data):
        pass

    def mppi_viz(self, goal, data, command, sub_time_s):
        sorted_traj_indices = np.argsort(self.mppi.cost)

        # viz
        i = None
        for i in range(min(self.mppi.num_samples, 10)):
            sorted_traj_idx = sorted_traj_indices[i]
            cost_normalized = self.mppi.cost_normalized[sorted_traj_idx]
            left_tool_pos, right_tool_pos = goal.tool_positions(self.mppi.rollout_results)
            left_tool_pos = left_tool_pos[sorted_traj_idx]
            right_tool_pos = right_tool_pos[sorted_traj_idx]
            c = cm.RdYlGn(1 - cost_normalized)
            self.viz.lines(left_tool_pos, ns='left_ee', idx=i, scale=0.002, color=c)
            self.viz.lines(right_tool_pos, ns='right_ee', idx=i, scale=0.002, color=c)
            rospy.sleep(0.01)

        cmd_rollout_results = rollout(self.model, copy(data), command[None], sub_time_s,
                                      get_result_func=goal.get_results)
        left_tool_pos, right_tool_pos = goal.tool_positions(cmd_rollout_results)
        self.viz.lines(left_tool_pos, ns='left_ee', idx=i, scale=0.004, color='b')
        self.viz.lines(right_tool_pos, ns='right_ee', idx=i, scale=0.004, color='b')
