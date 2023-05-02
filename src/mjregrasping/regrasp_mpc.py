import logging
from copy import copy

import numpy as np
import rerun as rr
from colorama import Fore
from matplotlib import cm

import rospy
from mjregrasping.buffer import Buffer
from mjregrasping.goals import ObjectPointGoal, GripperPointGoal, BodyWithChildren
from mjregrasping.grasping import get_grasp_indices, get_grasp_constraints
from mjregrasping.mujoco_mppi import MujocoMPPI
from mjregrasping.mujoco_visualizer import plot_lines_rviz, MujocoVisualizer
from mjregrasping.params import params
from mjregrasping.rollout import control_step, rollout

logger = logging.getLogger(f'rosout.{__name__}')



class RegraspMPC:

    def __init__(self, model, mjviz: MujocoVisualizer, viz_pubs, pool):
        self.model = model
        self.viz_pubs = viz_pubs
        self.mjviz = mjviz
        self.rope = BodyWithChildren(model, 'rope')
        self.goal = ObjectPointGoal(model=self.model,
                                    viz_pubs=viz_pubs,
                                    goal_point=np.array([0.85, 0.04, 1.27]),
                                    body_idx=-1,
                                    goal_radius=0.05)

        n_samples = params['move_to_goal']['n_samples']
        horizon = params['move_to_goal']['horizon']
        lambda_ = params['move_to_goal']['lambda']
        self.mppi = MujocoMPPI(pool, self.model, num_samples=n_samples, noise_sigma=np.deg2rad(10), horizon=horizon,
                               lambda_=lambda_)
        self.buffer = Buffer(3)

    def run(self, data):
        for i in range(params['iters']):
            self.move_to_goal(data)
            self.regrasp(data)

    def move_to_goal(self, data):
        """
        Runs MPPI with our novel cost function until the goal is reached or the robot is struck
        """
        warmstart_count = 0
        while True:
            self.goal.viz()
            if self.goal.satisfied(data):
                logger.info(Fore.GREEN + "Goal reached!" + Fore.RESET)
                return

            while warmstart_count < params['warmstart']:
                command = self.mppi.command(data, self.goal.get_results, self.goal.cost)
                self.mppi_viz(self.goal, data, command)
                warmstart_count += 1
            command = self.mppi.command(data, self.goal.get_results, self.goal.cost)
            self.mppi_viz(self.goal, data, command)

            self.mjviz.viz(self.model, data)
            control_step(self.model, data, command)

            self.mppi.roll()

            needs_regrasp = self.check_needs_regrasp(data)
            if needs_regrasp:
                break

    def check_needs_regrasp(self, data):
        external_force = data.qfrc_actuator - data.qfrc_bias - data.qfrc_passive
        max_force = np.max(external_force)
        total_force = np.sum(np.abs(external_force))
        self.buffer.insert(total_force)
        mean_total_force = np.mean(self.buffer.data)
        rr.log_scalar("external_force/max", max_force)
        rr.log_scalar("external_force/total", total_force)
        rr.log_scalar("external_force/mean_total", mean_total_force)
        return mean_total_force > 120

    def regrasp(self, data):
        # new_grasp = self.compute_new_grasp(data)
        current_grasp = get_grasp_indices(self.model)
        new_grasp = np.array([56, -1])
        if np.all(new_grasp == current_grasp):
            logger.info(Fore.GREEN + "No need to regrasp!" + Fore.RESET)
            return
        elif np.all(new_grasp != current_grasp) and np.all(new_grasp != -1):
            raise NotImplementedError("Cannot regrasp with both grippers at once")
        elif new_grasp[0] != current_grasp[0]:
            gripper_idx = 0
        else:
            gripper_idx = 1

        rope_body_to_grasp = self.model.body(new_grasp[gripper_idx])
        # NOTE: why does body.pos give the wrong value?
        logger.info(f"grasping {rope_body_to_grasp.name}")

        warmstart_count = 0
        while True:
            # recreate the goal at each time step so the goal point is updated
            regrasp_goal = GripperPointGoal(model=self.model,
                                            goal_point=data.xpos[rope_body_to_grasp.id],
                                            goal_radius=0.01,
                                            gripper_idx=0,
                                            viz_pubs=self.viz_pubs)

            regrasp_goal.viz()
            if regrasp_goal.satisfied(data):
                logger.info(Fore.GREEN + "Regrasp successful!" + Fore.RESET)
                break

            while warmstart_count < params['warmstart']:
                command = self.mppi.command(data, regrasp_goal.get_results, regrasp_goal.cost)
                self.mppi_viz(regrasp_goal, data, command)
                warmstart_count += 1
            command = self.mppi.command(data, regrasp_goal.get_results, regrasp_goal.cost)
            self.mppi_viz(regrasp_goal, data, command)

            self.mjviz.viz(self.model, data)
            control_step(self.model, data, command)

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

    def mppi_viz(self, goal, data, command):
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
            plot_lines_rviz(self.viz_pubs.ee_path, left_tool_pos, label='left_ee', idx=i, scale=0.002, color=c)
            plot_lines_rviz(self.viz_pubs.ee_path, right_tool_pos, label='right_ee', idx=i, scale=0.002, color=c)
            rospy.sleep(0.01)

        cmd_rollout_results = rollout(self.model, copy(data), command[None], get_result_func=goal.get_results)
        left_tool_pos, right_tool_pos = goal.tool_positions(cmd_rollout_results)
        plot_lines_rviz(self.viz_pubs.ee_path, left_tool_pos, label='left_ee', idx=i, scale=0.004, color='b')
        plot_lines_rviz(self.viz_pubs.ee_path, right_tool_pos, label='right_ee', idx=i, scale=0.004, color='b')
