#!/usr/bin/env python3

import argparse
from copy import copy

import mujoco
import numpy as np
from matplotlib import cm

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.mujoco_mppi import MujocoMPPI
from mjregrasping.mujoco_visualizer import RVizPublishers, plot_sphere_rviz, MujocoVisualizer, plot_lines_rviz
from mjregrasping.rollout import control_step, rollout


def left_gripper_xpos(model, data):
    return data.site_xpos[model.site('left_tool').id]


def left_gripper_xpos_and_qpos(model, data):
    return data.site_xpos[model.site('left_tool').id], data.qpos


def main():
    rospy.init_node("sample_to_goal")

    parser = argparse.ArgumentParser()
    parser.add_argument("xml_path", type=str)

    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    horizon = 10
    n_samples = 100

    left_gripper_goal_point = np.array([0.5, 0.5, 0.5])
    tfw = TF2Wrapper()
    viz_pubs = RVizPublishers(tfw)
    mjviz = MujocoVisualizer(tfw)
    np.set_printoptions(precision=4, suppress=True)
    costs_viz = []

    def _cost_func(results):
        left_gripper_positions, _ = results
        return np.linalg.norm(left_gripper_positions[:, 1:] - left_gripper_goal_point, axis=-1)

    mppi = MujocoMPPI(model, num_samples=n_samples, noise_sigma=np.deg2rad(5), horizon=horizon, lambda_=0.05)

    for warmstart_i in range(10):
        command = mppi._command(data, left_gripper_xpos_and_qpos, _cost_func)
        viz(mppi, data, model, command, viz_pubs)

    for t in range(100):
        mjviz.viz(model, data)
        plot_sphere_rviz(viz_pubs.goal_markers_pub, left_gripper_goal_point, 0.01, label='goal')

        # warmstart
        command = mppi.command(data, left_gripper_xpos_and_qpos, _cost_func)
        costs_viz.append(mppi.cost)
        viz(mppi, data, model, command, viz_pubs)

        # actually step
        control_step(model, data, command)

    costs_viz = np.array(costs_viz)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.min(costs_viz, axis=1), label='min')
    plt.plot(np.mean(costs_viz, axis=1), label='mean')
    plt.plot(np.max(costs_viz, axis=1), label='max')
    plt.legend()
    plt.show()


def viz(mppi, data, model, command, viz_pubs):
    sorted_traj_indices = np.argsort(mppi.cost)

    # viz
    for i in range(min(mppi.num_samples, 10)):
        sorted_traj_idx = sorted_traj_indices[i]
        cost_normalized = mppi.cost_normalized[sorted_traj_idx]
        left_gripper_positions = mppi.rollout_results[0][sorted_traj_idx]
        c = cm.RdYlGn(1 - cost_normalized)
        plot_lines_rviz(viz_pubs.ee_path_pub, left_gripper_positions, label='ee', idx=i, scale=0.001, color=c)
    left_gripper_positions, _ = rollout(model, copy(data), command[None], get_result_func=left_gripper_xpos_and_qpos)
    plot_lines_rviz(viz_pubs.ee_path_pub, left_gripper_positions, label='command', scale=0.004, color='blue')


if __name__ == "__main__":
    main()
