import time

import numpy as np
import rerun as rr

from mjregrasping.goals import BaseRegraspGoal
from mjregrasping.grasp_state import GraspState
from mjregrasping.grasp_state_utils import grasp_locations_to_indices_and_offsets
from mjregrasping.grasping import gripper_idx_to_eq_name, deactivate_eq
from mjregrasping.mujoco_mppi import MujocoMPPI, softmax
from mjregrasping.params import hp
from mjregrasping.rerun_visualizer import log_line_with_std
from mjregrasping.rollout import get_result_tuple, control_step
from mjregrasping.settle import settle


class RegraspMPPI(MujocoMPPI):
    """ The regrasping problem has a slightly different action representation and rollout function """

    def __init__(self, pool, nu, seed, horizon, noise_sigma, n_g, rope_body_indices, temp=1.):
        # TODO: add an additional gripper action for releasing initially
        super().__init__(pool, nu, seed, horizon, noise_sigma, temp, None)
        self.rope_body_indices = rope_body_indices
        self.n_g = n_g  # number of grippers
        self.rng = np.random.RandomState(seed)
        self.u_sigma_diag = np.ones(nu) * noise_sigma
        self.u_mu = np.zeros([self.horizon * nu])
        # num release events, minus the one for the useless case where all grippers release
        self.n_release_events = 2 ** self.n_g - 1
        self.release_logits = np.zeros([self.n_release_events])
        self.release_logits[0] = 1  # prior of no releasing

    def reset(self):
        raise NotImplementedError()

    def roll(self):
        u_mu_square = self.u_mu.reshape(self.horizon, self.nu)
        u_mu_square[:-1] = u_mu_square[1:]
        u_mu_square[-1] = 0
        self.u_mu = u_mu_square.reshape(-1)
        self.release_logits[0] = 1  # reset prior of no releasing

    def command(self, phy, goal: BaseRegraspGoal, sub_time_s, num_samples, viz=None):
        u_sigma_diag_rep = np.tile(self.u_sigma_diag, self.horizon)
        u_sigma_mat = np.diagflat(u_sigma_diag_rep)

        u_samples = self.noise_rng.multivariate_normal(self.u_mu, u_sigma_mat, size=num_samples)

        lower = np.tile(phy.m.actuator_ctrlrange[:, 0], self.horizon)
        upper = np.tile(phy.m.actuator_ctrlrange[:, 1], self.horizon)
        u_samples = np.clip(u_samples, lower, upper)

        # # Test a hand-designed trajectory to see if my cost function is behaving correctly
        # perturbed_action[0, 0] = 1
        # perturbed_action[0, 1] = 1
        # grasp_repeats = np.arange(0, self.horizon) * self.nu
        # move_repeats = np.arange(0, self.horizon) * self.nu + self.horizon * self.nu
        # perturbed_action[0, 2:] = 0
        # perturbed_action[0, 3 + grasp_repeats] = 0.26  # bend forward
        # perturbed_action[0, 4 + grasp_repeats] = 0.28  # left shoulder forward
        # perturbed_action[0, 5 + grasp_repeats] = -0.2  # left shoulder in
        # perturbed_action[0, 7 + grasp_repeats] = -0.23  # left elbow
        # perturbed_action[0, 3 + move_repeats] = -0.4  # reverse to move to goal
        # perturbed_action[0, 4 + move_repeats] = -0.4
        # perturbed_action[0, 7 + move_repeats] = 0.4

        u_noise = u_samples - self.u_mu

        current_grasp = GraspState.from_mujoco(self.rope_body_indices, phy.m)
        can_release = np.all(current_grasp.is_grasping)
        # Only if both grippers are grasping do we consider releasing
        # NOTE: hard-coded for <=2 grippers
        if can_release:
            pvals = softmax(self.release_logits, 1)
            release = self.noise_rng.multinomial(1, pvals)
        else:
            release = [None] * num_samples

        self.rollout_results, self.cost = self.regrasp_parallel_rollout(phy, goal, release, u_samples, num_samples,
                                                                        sub_time_s, viz=None)

        is_grasping = [[r_t[5] for r_t in results] for results in self.rollout_results]
        no_gripper_grasping = np.any(np.all(np.logical_not(is_grasping), axis=-1), axis=-1)
        ever_not_grasping_cost = no_gripper_grasping * hp['ever_not_grasping']
        self.cost += ever_not_grasping_cost

        log_line_with_std('regrasp_cost', self.cost, color=[0, 255, 255])

        # normalized cost is only used for visualization, so we avoid numerical issues
        cost_range = (self.cost.max() - self.cost.min())
        if cost_range < 1e-6:
            cost_range = 1.0
        self.cost_normalized = (self.cost - self.cost.min()) / cost_range

        weights = softmax(-self.cost, self.temp)
        print(f'weights: std={float(np.std(weights)):.3f} max={float(np.max(weights)):.2f}')
        rr.log_tensor('weights', weights)
        rr.log_tensor('σ', self.u_sigma_diag)
        if can_release:
            rr.log_tensor('relase pvals', pvals)
            rr.log_tensor('release', release)

        # compute the (weighted) average noise and add that to the reference control
        weighted_avg_noise = np.sum(weights[:, None] * u_noise, axis=0)

        # Covariance update for u
        u_samples_square = u_samples.reshape(num_samples, self.horizon, self.nu)
        u_mu_square = self.u_mu.reshape(self.horizon, self.nu)
        self.u_sigma_diag = weights @ np.mean((u_samples_square - u_mu_square) ** 2, axis=1)

        self.u_mu += weighted_avg_noise
        if can_release:
            raise NotImplementedError()
            # TODO: update the logits of the multinomial distribution?
            self.release_logits = weights * self.release_logits

        command = u_mu_square[0]
        if can_release:
            raise NotImplementedError()
            release_command = 0  # TODO: sample from the updated multinomial again?
        else:
            release_command = None
        return release_command, command

    def sample_change_grasp(self, gripper_probabilities):
        r = self.rng.uniform(0, 1, self.n_g)
        gripper_action = r < gripper_probabilities
        return gripper_action

    def regrasp_parallel_rollout(self, phy, goal, release, u_samples, num_samples, sub_time_s, viz=None):
        u_samples_square = u_samples.reshape(num_samples, self.horizon, self.nu)

        # We must also copy model here because EQs are going to be changing
        args_sets = [(phy.copy_all(), self.rope_body_indices, goal, sub_time_s, *args_i) for args_i in
                     zip(release, u_samples_square)]

        if viz:
            results = []
            costs = []
            for args in args_sets:
                results_i, cost_i = regrasp_rollout(*args, viz)
                results.append(results_i)
                costs.append(cost_i)
        else:
            futures = [self.pool.submit(regrasp_rollout, *args) for args in args_sets]
            results = []
            costs = []
            for f in futures:
                results_i, cost_i = f.result()
                results.append(results_i)
                costs.append(cost_i)

        costs = np.array(costs)

        return results, costs


def regrasp_rollout(phy, rope_body_indices, goal, sub_time_s, release, u_sample, viz=None):
    """ Must be a free function, since it's used in a multiprocessing pool. All arguments must be picklable. """
    if viz:
        viz.viz(phy, is_planning=True)

    if release is None:
        pass
    else:
        if len(release) != 3:
            raise NotImplementedError(f"Not sure how to interpret {release=}")
        if release[0]:
            pass  # no release
        elif release[1]:
            deactivate_eq(phy.m, 'left')
            settle(phy, sub_time_s, viz=viz, is_planning=True)
        elif release[2]:
            deactivate_eq(phy.m, 'right')
            settle(phy, sub_time_s, viz=viz, is_planning=True)

    results = [get_result_tuple(goal.get_results, phy)]
    costs = []
    t_new_grasp = u_sample.shape[0]
    for t, u in enumerate(u_sample):
        time.sleep(0.01)
        control_step(phy, u, sub_time_s=sub_time_s)
        if viz:
            viz.viz(phy, is_planning=True)
        result_tuple = get_result_tuple(goal.get_results, phy)

        if t <= t_new_grasp:
            cost = goal.grasp_cost(result_tuple)
        else:
            cost = goal.move_cost(result_tuple)

        results.append(result_tuple)
        costs.append(cost)

        left_tool_pos, right_tool_pos = result_tuple[:2]

        for i, tool_pos in enumerate([left_tool_pos, right_tool_pos]):
            name = gripper_idx_to_eq_name(i)
            eq = phy.m.eq(name)
            is_grasping = bool(eq.active)
            if is_grasping:
                continue
            t_new_grasp = grasp_if_close(name, phy, rope_body_indices, t, t_new_grasp, tool_pos)

    # receding horizon cost
    gammas = np.power(0.9, np.arange(u_sample.shape[0]))
    per_time_cost = np.dot(gammas, costs)

    smoothness_costs = np.linalg.norm(u_sample[1:] - u_sample[:-1], axis=-1)
    smoothness_cost = np.dot(gammas[:-1], smoothness_costs) * hp['smoothness_weight']

    cost = per_time_cost + smoothness_cost

    return results, cost


def grasp_if_close(name, phy, rope_body_indices, t, t_new_grasp, tool_pos):
    # compute the loc [0, 1] of the closest point on the rope to the gripper
    # to do this, finely discretize into a piecewise linear function that maps loc ∈ [0,1] to R^3
    # then find the loc that minimizes the distance to the gripper
    best_d = np.inf
    best_idx = None
    best_offset = None
    for loc in np.linspace(0, 1, 100):
        body_idx, offset = grasp_locations_to_indices_and_offsets(loc, rope_body_indices)
        pos = phy.d.xpos[body_idx]
        d = np.linalg.norm(tool_pos - pos)
        if d < best_d:
            best_d = d
            best_idx = body_idx
            best_offset = offset
    # if we're close enough, activate the grasp constraint
    if best_d < 2 * hp["grasp_goal_radius"]:
        grasp_eq = phy.m.eq(name)
        grasp_eq.obj2id = best_idx
        grasp_eq.active = 1
        grasp_eq.data[3:6] = best_offset
        t_new_grasp = t
    return t_new_grasp
