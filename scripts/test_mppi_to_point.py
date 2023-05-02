#!/usr/bin/env python3

import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from time import perf_counter

import numpy as np
from matplotlib import cm

from mjregrasping.get_result_functions import get_left_tool_pos_and_contact_cost
from mjregrasping.initialize import initialize
from mjregrasping.mujoco_mppi import MujocoMPPI
from mjregrasping.mujoco_visualizer import plot_sphere_rviz, plot_lines_rviz
from mjregrasping.rollout import control_step, rollout


def main():
    np.set_printoptions(precision=4, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("xml_path", type=str)

    args = parser.parse_args()

    model, data, mjviz, viz_pubs = initialize("test_mppi", args.xml_path)

    horizon = 10
    n_samples = 50

    left_gripper_goal_point = np.array([1, 0.5, 0.6])
    costs_viz = []

    def _cost_func(results):
        left_gripper_positions, contact_cost = results
        gripper_pos_cost = np.linalg.norm(left_gripper_positions - left_gripper_goal_point, axis=-1)
        state_costs = gripper_pos_cost + contact_cost
        return state_costs[:, 1:]

    dts = []
    with ThreadPoolExecutor(multiprocessing.cpu_count()) as pool:
        mppi = MujocoMPPI(pool, model, num_samples=n_samples, noise_sigma=np.deg2rad(10), horizon=horizon,
                          lambda_=0.005)

        for warmstart_i in range(5):
            command = mppi.command(data, get_left_tool_pos_and_contact_cost, _cost_func)
            viz(mppi, data, model, command, viz_pubs)
            plot_sphere_rviz(viz_pubs.goal, left_gripper_goal_point, 0.01, label='goal')

        for t in range(10):
            mjviz.viz(model, data)
            plot_sphere_rviz(viz_pubs.goal, left_gripper_goal_point, 0.01, label='goal')

            # warmstart
            t0 = perf_counter()
            command = mppi.roll_and_command(data, get_left_tool_pos_and_contact_cost, _cost_func)
            dt = perf_counter() - t0
            dts.append(dt)
            print(f"mppi.command: {dt:.3f}s")
            costs_viz.append(mppi.cost)
            viz(mppi, data, model, command, viz_pubs)

            # actually step
            control_step(model, data, command)

    print(f"mean dt: {np.mean(dts):.3f}s")

    costs_viz = np.array(costs_viz)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(np.min(costs_viz, axis=1), label='min')
    # plt.plot(np.mean(costs_viz, axis=1), label='mean')
    # plt.plot(np.max(costs_viz, axis=1), label='max')
    # plt.legend()
    # plt.show()


def viz(mppi, data, model, command, viz_pubs):
    sorted_traj_indices = np.argsort(mppi.cost)

    # viz
    for i in range(min(mppi.num_samples, 10)):
        sorted_traj_idx = sorted_traj_indices[i]
        cost_normalized = mppi.cost_normalized[sorted_traj_idx]
        left_gripper_positions = mppi.rollout_results[0][sorted_traj_idx]
        c = cm.RdYlGn(1 - cost_normalized)
        plot_lines_rviz(viz_pubs.ee_path, left_gripper_positions, label='ee', idx=i, scale=0.001, color=c)
    left_gripper_positions, _ = rollout(model, copy(data), command[None],
                                        get_result_func=get_left_tool_pos_and_contact_cost)
    plot_lines_rviz(viz_pubs.ee_path, left_gripper_positions, label='command', scale=0.004, color='blue')


if __name__ == "__main__":
    main()
