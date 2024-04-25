import gc
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import numpy as np
import rerun as rr
from matplotlib import cm
from numpy.linalg import norm
import colorednoise

import rospy
from mjregrasping.goal_funcs import get_tool_points
from mjregrasping.goals import MPPIGoal
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.grasping import get_grasp_eqs, get_finger_qs, activate_grasp
from mjregrasping.math import softmax
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.rollout import control_step

hp['min_sub_time_s'] = 0.04
hp['max_sub_time_s'] = 0.04

class RegraspiCEM:

    def __init__(self, pool, nu, seed, horizon, noise_sigma, temp, lower, upper,
                 u_per_command=1, num_elites=10, noise_beta=2,
                 elites_keep_fraction=.5):
        self.pool = pool
        self.horizon = horizon
        self.nu = nu

        self.initial_noise_sigma = noise_sigma
        self.seed = seed
        self.temp = temp

        self.noise_rng = np.random.RandomState(seed)
        self.u_sigma_diag = np.ones(nu * self.horizon) * self.initial_noise_sigma
        self.zero_grippers_sigma()
        self.u_mu = np.zeros([self.horizon * nu])
        self.time_sigma = 0.03
        self.time_mu = hp['sub_time_s']

        # sampled results from last command
        self.rollout_results = None
        self.cost = None
        self.cost_normalized = None
        self.lower = lower
        self.upper = upper

        self.u_per_command = u_per_command
        self.num_elites = num_elites
        self.keep_fraction = elites_keep_fraction
        self.kept_elites = None

        self.noise_beta = noise_beta

    def zero_grippers_sigma(self):
        # self.u_sigma_diag[9] = 0
        # self.u_sigma_diag[17] = 0
        pass

    def reset(self):
        self.u_sigma_diag = np.ones(self.nu * self.horizon) * self.initial_noise_sigma
        self.zero_grippers_sigma()
        self.u_mu = np.zeros([self.horizon * self.nu])
        self.time_mu = hp['sub_time_s']

    def roll(self):
        u_mu_square = self.u_mu.reshape(self.horizon, self.nu)
        u_mu_square[:-1] = u_mu_square[1:]
        # u_mu_square[-1] = u_mu_square[-2]  # repeat last action
        u_mu_square[-1] = 0  # just using 0 is another reasonable choice
        self.u_mu = u_mu_square.reshape(-1)
        self.u_sigma_diag = np.ones(self.nu * self.horizon) * self.initial_noise_sigma

        # u_sigma_square = self.u_sigma_diag.reshape(self.horizon, self.nu)
        # u_sigma_square[:-1] = u_sigma_square[1:]
        # u_sigma_square[-1] = self.initial_noise_sigma
        # self.u_sigma_diag = u_sigma_square.reshape(-1)

        #Roll elites
        if self.kept_elites is not None:
            self.kept_elites = np.roll(self.kept_elites, -self.nu, axis=1)
            self.kept_elites[:, -self.nu:] = self.initial_noise_sigma * self.noise_rng.randn(self.kept_elites.shape[0], self.nu)

    def command(self, phy, goal, num_samples, viz=None, iters=1):
        for _ in range(iters):
            sample_adj = 0
            if self.kept_elites is not None:
                sample_adj = len(self.kept_elites)
            if self.noise_beta > 0:
                perturbations = colorednoise.powerlaw_psd_gaussian(
                    self.noise_beta, size=(num_samples - sample_adj, self.nu, self.horizon)).transpose((0, 2, 1))
            else:
                perturbations = self.noise_rng.randn(num_samples - sample_adj, self.horizon, self.nu)
            perturbations = perturbations.reshape(-1, self.horizon * self.nu)
            u_samples = self.u_mu + perturbations * self.u_sigma_diag

            if self.kept_elites is not None:
                u_samples = np.concatenate((u_samples, self.kept_elites), axis=0)
            


            time_samples = self.noise_rng.normal(self.time_mu, self.time_sigma, size=num_samples)

            # Bound u
            lower = np.tile(self.lower, self.horizon)
            upper = np.tile(self.upper, self.horizon)
            u_samples = np.clip(u_samples, lower, upper)

            print(self.u_sigma_diag)
            print(perturbations)
            print(u_samples)

            #Add mean to samples
            # u_samples[0] = self.u_mu

            self.u_samples = u_samples.reshape(num_samples, self.horizon, self.nu)

            # Also bound time
            time_samples = np.clip(time_samples, hp['min_sub_time_s'], hp['max_sub_time_s'])

            u_noise = u_samples - self.u_mu
            self.u_noise = u_noise
            time_noise = time_samples - self.time_mu

            self.rollout_results, self.cost, costs_by_term = parallel_rollout(self.pool,
                                                                            self.horizon,
                                                                            self.nu,
                                                                            phy, goal, u_samples,
                                                                            time_samples,
                                                                            num_samples,
                                                                            viz=None)
            self.cost_range = (self.cost.max() - self.cost.min())
            if self.cost_range < 1e-6:
                self.cost_range = 1.0
            self.cost_normalized = (self.cost - self.cost.min()) / self.cost_range

            elites = np.argsort(self.cost)[:self.num_elites]
            elite_u = u_samples[elites]

            self.u_mu = .5 * self.u_mu + .5 * np.mean(elite_u, axis=0)
            self.u_sigma_diag = .5 * self.u_sigma_diag + .5 * np.std(elite_u, axis=0)

            #Keep previous elites
            self.kept_elites = u_samples[elites[:int(self.num_elites * self.keep_fraction)]]

        if viz is not None:
            rr.log_scalar("time μ", self.time_mu)

        #Execute the best command
        new_u_square = elite_u[0].reshape(self.horizon, self.nu)
        self.U = new_u_square


        if self.u_per_command == 1:
            command = new_u_square[0]
        else:
            command = new_u_square[:self.u_per_command].copy()
        return command, self.time_mu

parallel_phys = None

def parallel_rollout(pool, horizon, nu, phy, goal, u_samples, time_samples, num_samples, viz):
    u_samples_square = u_samples.reshape(num_samples, horizon, nu)
    # u_samples_square[..., [2, 5]] = .01
    # We must also copy model here because EQs are going to be changing
    num_serial = 1#int(num_samples/pool._max_workers)
    num_submits = int(num_samples/num_serial)
    global parallel_phys
    if parallel_phys is None:
        parallel_phys = [phy.copy_all() for _ in range(num_submits)]
    args_sets = []
    for i in range(num_submits):
        args_sets.append((goal, parallel_phys[i], phy.get_state(), u_samples_square[i * num_serial: (i+1)* num_serial], time_samples[i * num_serial: (i+1)* num_serial]))

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
            results.extend(results_i)
            costs.extend(cost_i)
            costs_by_term.extend(costs_i_by_term)

    results = np.stack(results, dtype=object, axis=1)
    costs = np.stack(costs, axis=0)

    costs_by_term = np.stack(costs_by_term, axis=0)

    return results, costs, costs_by_term

def rollout(goal, parallel_phy, state, u_samples, sub_time_ss, viz=None):
    """ Must be a free function, since it's used in a multiprocessing pool. All arguments must be picklable. """
    all_results = []
    all_costs = []
    all_costs_by_term = []
    for serial in range(len(u_samples)):
        u_sample = u_samples[serial]
        sub_time_s = sub_time_ss[serial]
        parallel_phy.set_state(*state)
        if viz:
            viz.viz(parallel_phy, is_planning=True)

        results_0 = goal.get_results(parallel_phy)
        # Only do this at the beginning, since it's expensive and if it went in the loop, it could potentially cause
        # rapid oscillations of grasping/not grasping which seems undesirable.
        # do_grasp_dynamics(phy)
        results = [results_0]
        for t, u in enumerate(u_sample):
            sim_crash = False
            try:
                control_step(parallel_phy, u, sub_time_s=sub_time_s)
            except Exception as e:
                sim_crash=True
            if viz:
                time.sleep(0.01)
                viz.viz(parallel_phy, is_planning=True)
            results_t = goal.get_results(parallel_phy, sim_crash)

            results.append(results_t)
        if sim_crash:
            print('Simulation crashed')
        results = np.stack(results, dtype=object, axis=1)

        costs_by_term = goal.costs(results, u_sample)  # ignore cost of initial state, it doesn't matter for planning

        cost = sum(costs_by_term)

        all_results.append(results)
        all_costs.append(cost)
        all_costs_by_term.append(costs_by_term)

    return all_results, all_costs, all_costs_by_term


def do_grasp_dynamics(phy: Physics, val_cmd = None):
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


def mppi_viz(mppi: RegraspiCEM, goal: MPPIGoal, phy: Physics, command: np.ndarray, sub_time_s: float):
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

