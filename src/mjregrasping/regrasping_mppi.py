import time
from typing import Optional

import numpy as np
import rerun as rr
from matplotlib import cm
from numpy.linalg import norm

import rospy
from mjregrasping.goal_funcs import get_tool_points
from mjregrasping.goals import MPPIGoal
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.grasping import get_grasp_eqs, get_finger_qs, activate_grasp
from mjregrasping.math import softmax
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.real_val import RealValCommander
from mjregrasping.rollout import control_step


class RegraspMPPI:

    def __init__(self, pool, nu, seed, horizon, noise_sigma, temp):
        self.pool = pool
        self.horizon = horizon
        self.nu = nu

        self.initial_noise_sigma = noise_sigma
        self.seed = seed
        self.temp = temp

        self.noise_rng = np.random.RandomState(seed)
        self.u_sigma_diag = np.ones(nu) * self.initial_noise_sigma
        self.u_mu = np.zeros([self.horizon * nu])
        self.time_sigma = 0.03
        self.time_mu = hp['sub_time_s']

        # sampled results from last command
        self.rollout_results = None
        self.cost = None
        self.cost_normalized = None

    def reset(self):
        self.u_sigma_diag = np.ones(self.nu) * self.initial_noise_sigma
        self.u_mu = np.zeros([self.horizon * self.nu])
        self.time_mu = hp['sub_time_s']

    def roll(self):
        u_mu_square = self.u_mu.reshape(self.horizon, self.nu)
        u_mu_square[:-1] = u_mu_square[1:]
        # u_mu_square[-1] = u_mu_square[-2]  # repeat last action
        u_mu_square[-1] = 0  # just using 0 is another reasonable choice
        self.u_mu = u_mu_square.reshape(-1)

    def command(self, phy, goal, num_samples, viz=None):
        u_sigma_diag_rep = np.tile(self.u_sigma_diag, self.horizon)
        u_sigma_mat = np.diagflat(u_sigma_diag_rep)

        u_samples = self.noise_rng.multivariate_normal(self.u_mu, u_sigma_mat, size=num_samples)
        time_samples = self.noise_rng.normal(self.time_mu, self.time_sigma, size=num_samples)

        # Bound u
        lower = np.tile(phy.m.actuator_ctrlrange[:, 0], self.horizon)
        upper = np.tile(phy.m.actuator_ctrlrange[:, 1], self.horizon)
        u_samples = np.clip(u_samples, lower, upper)
        # Also bound time
        time_samples = np.clip(time_samples, hp['min_sub_time_s'], hp['max_sub_time_s'])

        u_noise = u_samples - self.u_mu
        time_noise = time_samples - self.time_mu

        self.rollout_results, self.cost, costs_by_term = parallel_rollout(self.pool,
                                                                          self.horizon,
                                                                          self.nu,
                                                                          phy, goal, u_samples,
                                                                          time_samples,
                                                                          num_samples,
                                                                          viz=None)

        for cost_term_name, costs_for_term in zip(goal.cost_names(), costs_by_term.T):
            rr.log_scalar(f'mpc_costs/{cost_term_name}', np.mean(costs_for_term))

        # normalized cost is only used for visualization, so we avoid numerical issues
        cost_range = (self.cost.max() - self.cost.min())
        if cost_range < 1e-6:
            cost_range = 1.0
        self.cost_normalized = (self.cost - self.cost.min()) / cost_range

        weights = softmax(-self.cost, self.temp)
        # print(f'weights: std={float(np.std(weights)):.3f} max={float(np.max(weights)):.2f}')
        rr.log_tensor('weights', weights)
        rr.log_tensor('σ', self.u_sigma_diag)

        # compute the (weighted) average noise and add that to the reference control
        weighted_avg_u_noise = np.sum(weights[..., None] * u_noise, axis=0)
        weight_avg_time_noise = np.sum(weights * time_noise, axis=0)

        u_mu_square = self.u_mu.reshape(self.horizon, self.nu)

        # Update covariance
        u_samples_square = u_samples.reshape(num_samples, self.horizon, self.nu)
        self.u_sigma_diag = weights @ np.mean((u_samples_square - u_mu_square) ** 2, axis=1)
        # NOTE: we could adapt time_sigma with this: weights @ (time_samples - self.time_mu) ** 2

        # Update mean
        self.u_mu += weighted_avg_u_noise
        self.time_mu += weight_avg_time_noise

        rr.log_scalar("time μ", self.time_mu)

        new_u_mu_square = self.u_mu.reshape(self.horizon, self.nu)
        command = new_u_mu_square[0]
        return command, self.time_mu


def parallel_rollout(pool, horizon, nu, phy, goal, u_samples, time_samples, num_samples, viz):
    u_samples_square = u_samples.reshape(num_samples, horizon, nu)

    # We must also copy model here because EQs are going to be changing
    args_sets = [(phy.copy_all(), goal, *args_i) for args_i in zip(u_samples_square, time_samples)]

    if viz:
        results = []
        costs = []
        costs_by_term = []
        for args in args_sets:
            results_i, cost_i, costs_i_by_term = rollout(*args, viz)
            results.append(results_i)
            costs.append(cost_i)
            costs_by_term.append(costs_i_by_term)
    else:
        futures = [pool.submit(rollout, *args) for args in args_sets]
        results = []
        costs = []
        costs_by_term = []
        for f in futures:
            results_i, cost_i, costs_i_by_term = f.result()
            results.append(results_i)
            costs.append(cost_i)
            costs_by_term.append(costs_i_by_term)

    results = np.stack(results, dtype=object, axis=1)
    costs = np.stack(costs, axis=0)
    costs_by_term = np.stack(costs_by_term, axis=0)

    return results, costs, costs_by_term


def rollout(phy, goal, u_sample, sub_time_s, viz=None):
    """ Must be a free function, since it's used in a multiprocessing pool. All arguments must be picklable. """
    if viz:
        viz.viz(phy, is_planning=True)

    results_0 = goal.get_results(phy)
    # Only do this at the beginning, since it's expensive and if it went in the loop, it could potentially cause
    # rapid oscillations of grasping/not grasping which seems undesirable.
    do_grasp_dynamics(phy)
    results = [results_0]

    for t, u in enumerate(u_sample):
        control_step(phy, u, sub_time_s=sub_time_s)
        if viz:
            time.sleep(0.01)
            viz.viz(phy, is_planning=True)
        results_t = goal.get_results(phy)

        results.append(results_t)

    results = np.stack(results, dtype=object, axis=1)
    costs_by_term = goal.costs(results, u_sample)  # ignore cost of initial state, it doesn't matter for planning

    cost = sum(costs_by_term)

    return results, cost, costs_by_term


def do_grasp_dynamics(phy: Physics, val_cmd: Optional[RealValCommander] = None):
    tools_pos = get_tool_points(phy)
    finger_qs = get_finger_qs(phy)
    # NOTE: this function must be VERY fast, since we run it inside rollout() in a tight loop
    did_new_grasp = False
    eqs = get_grasp_eqs(phy)
    for tool_pos, finger_q, eq in zip(tools_pos, finger_qs, eqs):
        is_grasping = bool(eq.active)
        if is_grasping:
            # if the finger is open, release
            if finger_q > hp['finger_q_open']:
                eq.active = 0
                did_new_grasp = True
                if val_cmd:
                    val_cmd.set_cdcpd_grippers(phy)
        else:
            # compute the loc [0, 1] of the closest point on the rope to the gripper
            # to do this, finely discretize into a piecewise linear function that maps loc ∈ [0,1] to R^3
            # then find the loc that minimizes the distance to the gripper
            locs = np.linspace(0, 1, 25)
            body_idx, offset, xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy, locs)
            d = norm(tool_pos - xpos, axis=-1)
            best_idx = d.argmin()
            best_loc = locs[best_idx]
            best_d = d[best_idx]
            # if we're close enough and gripper angle is small enough, activate the grasp constraint
            if best_d < hp["grasp_goal_radius"] and abs(finger_q - hp['finger_q_closed']) < np.deg2rad(5):
                activate_grasp(phy, eq.name, best_loc)
                if val_cmd:
                    val_cmd.set_cdcpd_grippers(phy)
                did_new_grasp = True

    return did_new_grasp


def mppi_viz(mppi: RegraspMPPI, goal: MPPIGoal, phy: Physics, command: np.ndarray, sub_time_s: float):
    sorted_traj_indices = np.argsort(mppi.cost)

    i = None
    num_samples = mppi.cost.shape[0]
    for i in range(min(num_samples, 10)):
        sorted_traj_idx = sorted_traj_indices[i]
        cost_normalized = mppi.cost_normalized[sorted_traj_idx]
        c = list(cm.RdYlGn(1 - cost_normalized))
        c[-1] = 0.8
        result_i = mppi.rollout_results[:, sorted_traj_idx]
        goal.viz_result(phy, result_i, i, color=c, scale=0.002)
        rospy.sleep(0.001)  # needed otherwise messages get dropped :( I hate ROS...

    if command is not None:
        cmd_rollout_results, _, _ = rollout(phy.copy_all(), goal, np.expand_dims(command, 0),
                                            np.expand_dims(sub_time_s, 0), viz=None)
        goal.viz_result(phy, cmd_rollout_results, i, color='b', scale=0.004)
