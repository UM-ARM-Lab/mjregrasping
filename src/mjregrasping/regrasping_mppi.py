import time

import numpy as np
import rerun as rr
from numpy.linalg import norm

from mjregrasping.goals import as_float
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.grasping import get_grasp_eqs
from mjregrasping.mujoco_mppi import MujocoMPPI
from mjregrasping.math import softmax
from mjregrasping.params import hp
from mjregrasping.rerun_visualizer import log_line_with_std
from mjregrasping.rollout import control_step


class RegraspMPPI(MujocoMPPI):
    """ The regrasping problem has a slightly different action representation and rollout function """

    def __init__(self, pool, nu, seed, horizon, noise_sigma, n_g, temp=1.):
        # TODO: add an additional gripper action for releasing initially
        super().__init__(pool, nu, seed, horizon, noise_sigma, temp, None)
        self.n_g = n_g  # number of grippers
        self.rng = np.random.RandomState(seed)
        self.u_sigma_diag = np.ones(nu) * self.initial_noise_sigma
        self.u_mu = np.zeros([self.horizon * nu])

    def reset(self):
        self.u_sigma_diag = np.ones(self.nu) * self.initial_noise_sigma
        self.u_mu = np.zeros([self.horizon * self.nu])

    def roll(self):
        u_mu_square = self.u_mu.reshape(self.horizon, self.nu)
        u_mu_square[:-1] = u_mu_square[1:]
        u_mu_square[-1] = 0
        self.u_mu = u_mu_square.reshape(-1)

    def command(self, phy, goal, sub_time_s, num_samples, viz=None):
        u_sigma_diag_rep = np.tile(self.u_sigma_diag, self.horizon)
        u_sigma_mat = np.diagflat(u_sigma_diag_rep)

        u_samples = self.noise_rng.multivariate_normal(self.u_mu, u_sigma_mat, size=num_samples)

        lower = np.tile(phy.m.actuator_ctrlrange[:, 0], self.horizon)
        upper = np.tile(phy.m.actuator_ctrlrange[:, 1], self.horizon)
        u_samples = np.clip(u_samples, lower, upper)

        u_noise = u_samples - self.u_mu

        self.rollout_results, self.cost, costs_by_term = self.regrasp_parallel_rollout(phy, goal, u_samples,
                                                                                       num_samples,
                                                                                       sub_time_s,
                                                                                       viz=viz)

        cost_term_names = goal.cost_names() + ['smoothness', 'ever_not_grasping']
        for cost_term_name, costs_for_term in zip(cost_term_names, costs_by_term.T):
            rr.log_scalar(f'regrasp_goal/{cost_term_name}', np.mean(costs_for_term))

        log_line_with_std('regrasp_cost', self.cost, color=[0, 255, 255])

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
        weighted_avg_noise = np.sum(weights[..., None] * u_noise, axis=0)

        # Covariance update for u
        u_samples_square = u_samples.reshape(num_samples, self.horizon, self.nu)
        u_mu_square = self.u_mu.reshape(self.horizon, self.nu)
        self.u_sigma_diag = weights @ np.mean((u_samples_square - u_mu_square) ** 2, axis=1)

        self.u_mu += weighted_avg_noise

        command = u_mu_square[0]
        return command

    def regrasp_parallel_rollout(self, phy, goal, u_samples, num_samples, sub_time_s, viz):
        u_samples_square = u_samples.reshape(num_samples, self.horizon, self.nu)

        if not viz.p.viz_planning:
            viz = None

        # We must also copy model here because EQs are going to be changing
        args_sets = [(phy.copy_all(), goal, sub_time_s, *args_i) for args_i in zip(u_samples_square, )]

        if viz:
            results = []
            costs = []
            costs_by_term = []
            for args in args_sets:
                results_i, cost_i, costs_i_by_term = regrasp_rollout(*args, viz)
                results.append(results_i)
                costs.append(cost_i)
                costs_by_term.append(costs_i_by_term)
        else:
            futures = [self.pool.submit(regrasp_rollout, *args) for args in args_sets]
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


def regrasp_rollout(phy, goal, sub_time_s, u_sample, viz=None):
    """ Must be a free function, since it's used in a multiprocessing pool. All arguments must be picklable. """
    if viz:
        viz.viz(phy, is_planning=True)

    results_0 = goal.get_results(phy)
    do_grasp_dynamics(phy, results_0)
    results = [results_0]

    costs = []
    for t, u in enumerate(u_sample):
        control_step(phy, u, sub_time_s=sub_time_s)
        if viz:
            time.sleep(0.01)
            viz.viz(phy, is_planning=True)
        results_t = goal.get_results(phy)

        # If we haven't created a new grasp, we should add the grasp cost
        # This encourages grasping as quickly as possible
        costs_t = goal.cost(results_t)

        results.append(results_t)
        costs.append(costs_t)

    results = np.stack(results, dtype=object, axis=1)

    # receding horizon cost
    gammas = np.power(0.9, np.arange(u_sample.shape[0]))
    costs = np.stack(costs, axis=0)
    per_time_cost = np.dot(gammas, np.sum(costs, axis=-1))

    # FIXME: do a better smoothness cost that is more than just one time step, do the full correlation or something
    #  also how do we treat magnitude vs direction?
    u_diff_normalized = (u_sample[1:] - u_sample[:-1])
    smoothness_costs = norm(u_diff_normalized, axis=-1)
    smoothness_cost = np.dot(gammas[:-1], smoothness_costs) * hp['smoothness_weight']

    is_grasping = as_float(results[2])
    no_gripper_grasping = np.any(np.all(np.logical_not(is_grasping), axis=-1), axis=-1)
    ever_not_grasping_cost = no_gripper_grasping * hp['ever_not_grasping']

    cost = per_time_cost + smoothness_cost + ever_not_grasping_cost

    full_traj_costs = [smoothness_cost, ever_not_grasping_cost]
    costs_by_term = np.concatenate([costs.T @ gammas, full_traj_costs], 0)

    return results, cost, costs_by_term


def do_grasp_dynamics(phy, results):
    tools_pos = results[0]
    finger_qs = results[6]
    # NOTE: this function must be VERY fast, since we run it inside rollout() in a tight loop
    did_new_grasp = False
    eqs = get_grasp_eqs(phy.m)
    for tool_pos, finger_q, eq in zip(tools_pos, finger_qs, eqs):
        is_grasping = bool(eq.active)
        if is_grasping:
            # if the finger is open, release
            if finger_q > hp['finger_q_open']:
                eq.active = 0
                did_new_grasp = True
        else:
            # compute the loc [0, 1] of the closest point on the rope to the gripper
            # to do this, finely discretize into a piecewise linear function that maps loc ∈ [0,1] to R^3
            # then find the loc that minimizes the distance to the gripper
            locs = np.linspace(0, 1, 25)
            body_idx, offset, xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy, locs)
            d = norm(tool_pos - xpos, axis=-1)
            best_idx = d.argmin()
            best_body_idx = body_idx[best_idx]
            best_d = d[best_idx]
            best_offset = offset[best_idx]
            best_offset_body = np.array([best_offset, 0, 0])
            # if we're close enough and gripper angle is small enough, activate the grasp constraint
            if best_d < hp["grasp_goal_radius"] and finger_q < hp['finger_q_closed']:
                eq.obj2id = best_body_idx
                eq.active = 1
                eq.data[3:6] = best_offset_body
                did_new_grasp = True

    return did_new_grasp
