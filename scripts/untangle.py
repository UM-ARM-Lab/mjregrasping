#!/usr/bin/env python3
import importlib
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from time import perf_counter

import numpy as np
from matplotlib import cm

import rospy
from mjregrasping.goals import ObjectPointGoal
from mjregrasping.initialize import initialize, activate_eq
from mjregrasping.mppi_to_joint_config import pid_to_joint_config
from mjregrasping.mujoco_mppi import MujocoMPPI
from mjregrasping.mujoco_visualizer import plot_lines_rviz
from mjregrasping.rollout import control_step, rollout


def main():
    importlib.reload(logging)

    np.set_printoptions(precision=4, suppress=True)

    model, data, mjviz, viz_pubs = initialize("untangle", "models/untangle_scene.xml")

    setup_tangled_scene(model, data, mjviz)
    # setup_untangled_scene(model, data, mjviz)

    horizon = 9  # number of actions. states will be horizon + 1
    n_samples = 50

    goal = ObjectPointGoal(model=model,
                           viz_pubs=viz_pubs,
                           goal_point=np.array([0.85, 0.04, 1.27]),
                           body_idx=-1,
                           goal_radius=0.05)

    dts = []
    with ThreadPoolExecutor(multiprocessing.cpu_count() - 1) as pool:
        mppi = MujocoMPPI(pool, model, num_samples=n_samples, noise_sigma=np.deg2rad(10), horizon=horizon,
                          lambda_=0.01)

        for warmstart_i in range(5):
            command = mppi._command(data, goal.get_results, goal.cost)
            goal.viz()
            viz(mppi, goal, data, model, command, viz_pubs)

        for t in range(250):
            goal.viz()
            mjviz.viz(model, data)

            if goal.satisfied(data):
                print("goal satisfied")
                break

            t0 = perf_counter()
            command = mppi.command(data, goal.get_results, goal.cost)
            dt = perf_counter() - t0
            dts.append(dt)
            print(f"mppi.command: {dt:.3f}s")
            viz(mppi, goal, data, model, command, viz_pubs)

            # actually step
            control_step(model, data, command)

    print(f"mean dt: {np.mean(dts):.3f}s")


def viz(mppi, goal, data, model, command, viz_pubs):
    sorted_traj_indices = np.argsort(mppi.cost)

    # viz
    for i in range(min(mppi.num_samples, 10)):
        sorted_traj_idx = sorted_traj_indices[i]
        cost_normalized = mppi.cost_normalized[sorted_traj_idx]
        left_tool_pos, right_tool_pos = goal.tool_positions(mppi.rollout_results)
        left_tool_pos = left_tool_pos[sorted_traj_idx]
        right_tool_pos = right_tool_pos[sorted_traj_idx]
        c = cm.RdYlGn(1 - cost_normalized)
        plot_lines_rviz(viz_pubs.ee_path, left_tool_pos, label='left_ee', idx=i, scale=0.002, color=c)
        plot_lines_rviz(viz_pubs.ee_path, right_tool_pos, label='right_ee', idx=i, scale=0.002, color=c)
        rospy.sleep(0.01)

    cmd_rollout_results = rollout(model, copy(data), command[None], get_result_func=goal.get_results)
    left_tool_pos, right_tool_pos = goal.tool_positions(cmd_rollout_results)
    plot_lines_rviz(viz_pubs.ee_path, left_tool_pos, label='left_ee', idx=i, scale=0.004, color='b')
    plot_lines_rviz(viz_pubs.ee_path, right_tool_pos, label='right_ee', idx=i, scale=0.004, color='b')


def setup_tangled_scene(model, data, mjviz):
    robot_q1 = np.array([
        -0.7, 0.1,  # torso
        -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
        0, 0,  # left gripper
        0.0, -0.2, 0, -0.30, 0, -0.2, 0,  # right arm
        0, 0,  # right gripper
    ])
    pid_to_joint_config(mjviz, model, data, robot_q1)
    robot_q2 = np.array([
        -0.5, 0.4,  # torso
        -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
        0, 0,  # left gripper
        1.2, -0.2, 0, -0.90, 0, -0.2, 0,  # right arm
        0, 0,  # right gripper
    ])
    pid_to_joint_config(mjviz, model, data, robot_q2)
    activate_and_settle(data, mjviz, model)


def setup_untangled_scene(model, data, mjviz):
    activate_and_settle(data, mjviz, model)


def activate_and_settle(data, mjviz, model):
    # Activate the connect constraint between the rope and the gripper to
    activate_eq(model, 'right')
    # settle
    for _ in range(25):
        mjviz.viz(model, data)
        control_step(model, data, np.zeros(model.nu))
        rospy.sleep(0.01)


if __name__ == "__main__":
    main()
