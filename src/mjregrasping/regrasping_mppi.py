import time

import numpy as np
import rerun as rr

from mjregrasping.goals import as_float
from mjregrasping.grasp_state_utils import grasp_locations_to_indices_and_offsets
from mjregrasping.grasping import gripper_idx_to_eq_name, deactivate_eq, get_finger_qs
from mjregrasping.mujoco_mppi import MujocoMPPI, softmax
from mjregrasping.params import hp
from mjregrasping.rerun_visualizer import log_line_with_std
from mjregrasping.rollout import control_step


class RegraspMPPI(MujocoMPPI):
    """ The regrasping problem has a slightly different action representation and rollout function """

    def __init__(self, pool, nu, seed, horizon, noise_sigma, n_g, rope_body_indices, temp=1.):
        # TODO: add an additional gripper action for releasing initially
        super().__init__(pool, nu, seed, horizon, noise_sigma, temp, None)
        self.rope_body_indices = rope_body_indices
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

    def command(self, phy, goal, sub_time_s, num_samples, exploration_weight, viz=None):
        u_sigma_diag_rep = np.tile(self.u_sigma_diag, self.horizon)
        u_sigma_mat = np.diagflat(u_sigma_diag_rep)

        u_samples = self.noise_rng.multivariate_normal(self.u_mu, u_sigma_mat, size=num_samples)

        lower = np.tile(phy.m.actuator_ctrlrange[:, 0], self.horizon)
        upper = np.tile(phy.m.actuator_ctrlrange[:, 1], self.horizon)
        u_samples = np.clip(u_samples, lower, upper)

        u_noise = u_samples - self.u_mu

        self.rollout_results, self.cost, costs_by_term = self.regrasp_parallel_rollout(phy, goal, u_samples,
                                                                                       num_samples,
                                                                                       sub_time_s, exploration_weight,
                                                                                       viz=viz)

        cost_term_names = goal.move_cost_names() + ['smoothness', 'ever_not_grasping']
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

    def regrasp_parallel_rollout(self, phy, goal, u_samples, num_samples, sub_time_s, exploration_weight, viz):
        u_samples_square = u_samples.reshape(num_samples, self.horizon, self.nu)

        if not viz.p.viz_planning:
            viz = None

        # We must also copy model here because EQs are going to be changing
        args_sets = [(phy.copy_all(), self.rope_body_indices, goal, sub_time_s, exploration_weight, *args_i) for args_i
                     in
                     zip(u_samples_square, )]

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


def regrasp_rollout(phy, rope_body_indices, goal, sub_time_s, exploration_weight, u_sample, viz=None):
    """ Must be a free function, since it's used in a multiprocessing pool. All arguments must be picklable. """
    if viz:
        viz.viz(phy, is_planning=True)

    results_0 = goal.get_results(phy)
    left_tool_pos, right_tool_pos = results_0[:2]
    do_grasps_if_close(phy, left_tool_pos, right_tool_pos, rope_body_indices)
    release_dynamics(phy)
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
        costs_t = goal.cost(results_t, exploration_weight)

        results.append(results_t)
        costs.append(costs_t)

    results = np.stack(results, dtype=object, axis=1)

    # receding horizon cost
    gammas = np.power(0.9, np.arange(u_sample.shape[0]))
    costs = np.stack(costs, axis=0)
    per_time_cost = np.dot(gammas, np.sum(costs, axis=-1))

    smoothness_costs = np.linalg.norm(u_sample[1:] - u_sample[:-1], axis=-1)
    smoothness_cost = np.dot(gammas[:-1], smoothness_costs) * hp['smoothness_weight']

    is_grasping = as_float(results[3])
    no_gripper_grasping = np.any(np.all(np.logical_not(is_grasping), axis=-1), axis=-1)
    ever_not_grasping_cost = no_gripper_grasping * hp['ever_not_grasping']

    cost = per_time_cost + smoothness_cost + ever_not_grasping_cost

    full_traj_costs = [smoothness_cost, ever_not_grasping_cost]
    costs_by_term = np.concatenate([costs.T @ gammas, full_traj_costs], 0)

    return results, cost, costs_by_term


def do_grasps_if_close(phy, left_tool_pos, right_tool_pos, rope_body_indices):
    # NOTE: this function must be VERY fast, since we run it inside rollout() in a tight loop
    did_new_grasp = False
    for i, tool_pos in enumerate([left_tool_pos, right_tool_pos]):
        name = gripper_idx_to_eq_name(i)
        eq = phy.m.eq(name)
        is_grasping = bool(eq.active)
        if is_grasping:
            continue

        # compute the loc [0, 1] of the closest point on the rope to the gripper
        # to do this, finely discretize into a piecewise linear function that maps loc ∈ [0,1] to R^3
        # then find the loc that minimizes the distance to the gripper
        locs = np.linspace(0, 1, 25)
        body_idx, offset = grasp_locations_to_indices_and_offsets(locs, rope_body_indices)
        pos = phy.d.xpos[body_idx]
        d = np.linalg.norm(tool_pos - pos, axis=-1)
        best_idx = d.argmin()
        best_body_idx = body_idx[best_idx]
        best_d = d[best_idx]
        best_offset = offset[best_idx]
        offset_body = np.array([best_offset, 0, 0])
        # if we're close enough, activate the grasp constraint
        if best_d < hp["grasp_goal_radius"]:
            eq.obj2id = best_body_idx
            eq.active = 1
            eq.data[3:6] = offset_body
            print(f'{offset_body=}')
            did_new_grasp = True
    return did_new_grasp


def release_dynamics(phy):
    leftgripper_q, rightgripper_q = get_finger_qs(phy)
    left_eq = phy.m.eq('left')
    right_eq = phy.m.eq('right')
    did_release = False
    if leftgripper_q > hp['open_finger_q'] and bool(left_eq.active):
        left_eq.active = 0
        did_release = True
    if rightgripper_q > hp['open_finger_q'] and bool(right_eq.active):
        right_eq.active = 0
        did_release = True

    return did_release
