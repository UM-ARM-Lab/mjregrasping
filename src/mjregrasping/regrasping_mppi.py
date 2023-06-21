import numpy as np
import rerun as rr

from mjregrasping.goals import BaseRegraspGoal
from mjregrasping.grasp_state_utils import grasp_locations_to_indices_and_offsets
from mjregrasping.grasping import gripper_idx_to_eq_name, deactivate_eq
from mjregrasping.mujoco_mppi import MujocoMPPI, softmax
from mjregrasping.params import hp
from mjregrasping.rollout import get_result_tuple, control_step, list_of_tuples_to_tuple_of_lists
from mjregrasping.settle import settle


class RegraspMPPI(MujocoMPPI):
    """ The regrasping problem has a slightly different action representation and rollout function """

    def __init__(self, pool, nu, seed, horizon, noise_sigma, n_g, rope_body_indices, temp=1., gamma=0.9):
        # TODO: add an additional gripper action for releasing initially
        super().__init__(pool, nu, seed, horizon, noise_sigma, temp, gamma)
        self.rope_body_indices = rope_body_indices
        self.total_nu = n_g + 2 * horizon * nu
        self.n_g = n_g  # number of grippers
        self.rng = np.random.RandomState(seed)
        self.noise_sigma_diag = np.ones(self.n_g + nu) * noise_sigma
        self.noise_sigma_diag[:self.n_g] = 0.25  # this noise on a probability, so you shouldn't ever change it
        self.U = np.zeros([self.total_nu])
        self.U[:self.n_g] = 1  # prior says we want to change our grasp

    def reset(self):
        raise NotImplementedError()

    def roll(self):
        raise NotImplementedError()

    def command(self, phy, goal: BaseRegraspGoal, sub_time_s, num_samples, viz=None):
        gripper_sigma_diag = self.noise_sigma_diag[:self.n_g]
        nongripper_sigma_diag = self.noise_sigma_diag[self.n_g:]
        nongripper_sigma_diag_rep = np.repeat(nongripper_sigma_diag, 2 * self.horizon)
        sigma_diag = np.concatenate([gripper_sigma_diag, nongripper_sigma_diag_rep])  # [ n_g + 2 * horizon * nu]
        sigma_matrix = np.diagflat(sigma_diag)

        noise = self.noise_rng.randn(num_samples, self.total_nu) @ np.sqrt(sigma_matrix)
        perturbed_action = self.U + noise

        lower = np.tile(phy.m.actuator_ctrlrange[:, 0], 2 * self.horizon)
        lower = np.concatenate([np.zeros(self.n_g), lower])
        upper = np.tile(phy.m.actuator_ctrlrange[:, 1], 2 * self.horizon)
        upper = np.concatenate([np.ones(self.n_g), upper])

        # NOTE: We clip the absolute action, then recompute the noise
        #  so that later when we weight noise, we're weighting the bounded noise.
        perturbed_action = np.clip(perturbed_action, lower, upper)

        # Test a hand-designed trajectory to see if my cost function is behaving correctly
        perturbed_action[0, 0] = 1
        perturbed_action[0, 1] = 1
        grasp_repeats = np.arange(0, self.horizon) * self.nu
        move_repeats = np.arange(0, self.horizon) * self.nu + self.horizon * self.nu
        perturbed_action[0, 2:] = 0
        perturbed_action[0, 3 + grasp_repeats] = 0.26  # bend forward
        perturbed_action[0, 4 + grasp_repeats] = 0.28  # left shoulder forward
        perturbed_action[0, 5 + grasp_repeats] = -0.2  # left shoulder in
        perturbed_action[0, 7 + grasp_repeats] = -0.23  # left elbow
        perturbed_action[0, 3 + move_repeats] = -0.4  # reverse to move to goal
        perturbed_action[0, 4 + move_repeats] = -0.4
        perturbed_action[0, 7 + move_repeats] = 0.4

        noise = perturbed_action - self.U

        change_grasp, grasp_results, move_results, grasp_change_errors = self.regrasp_parallel_rollout(phy, goal,
                                                                                                       perturbed_action,
                                                                                                       num_samples,
                                                                                                       sub_time_s,
                                                                                                       viz=None)
        self.rollout_results = tuple([np.concatenate([g_r, m_r], 1) for g_r, m_r, in zip(grasp_results, move_results)])
        no_gripper_grasping = np.any(np.all(np.logical_not(self.rollout_results[5]), axis=-1), axis=-1)
        ever_not_grasping_cost = no_gripper_grasping * hp['ever_not_grasping']
        grasp_change_cost = grasp_change_errors ** 2 * hp['grasp_change_error']

        grasp_costs = goal.grasp_cost(grasp_results)
        move_costs = goal.move_cost(move_results)

        self.costs = np.concatenate([grasp_costs, move_costs], 1)
        # NOTE: by adding grasp_costs and move_costs, then multiplying by gamma in this way, we are discounting
        #  in time (as usual) but treating the start of the move costs as t=0,
        #  in the same way as the start of the grasp costs which is a little weird.
        gammas = np.power(self.gamma, np.arange(2 * self.horizon))[None]
        self.cost = np.sum(gammas * self.costs, axis=-1)
        self.cost += ever_not_grasping_cost
        self.cost += grasp_change_cost

        # normalized cost is only used for visualization, so we avoid numerical issues
        cost_range = (self.cost.max() - self.cost.min())
        if cost_range < 1e-6:
            cost_range = 1.0
        self.cost_normalized = (self.cost - self.cost.min()) / cost_range

        weights = softmax(-self.cost, self.temp)
        print(f'weights: std={float(np.std(weights)):.3f} max={float(np.max(weights)):.2f}')
        rr.log_tensor('weights', weights)
        rr.log_tensor('σ', self.noise_sigma_diag)

        # compute the (weighted) average noise and add that to the reference control
        weighted_avg_noise = np.sum(weights[:, None] * noise, axis=0)

        # Covariance matrix adaptation:
        U_gripper = self.U[:self.n_g]
        U_nongripper = self.U[self.n_g:]
        U_nongripper_square = U_nongripper.reshape(2 * self.horizon, self.nu)
        nongripper_action = perturbed_action[:, self.n_g:]
        nongripper_action_square = nongripper_action.reshape(num_samples, 2 * self.horizon, self.nu)
        gripper_sigma_diag = weights @ (U_gripper[None] - change_grasp) ** 2
        nongripper_sigma_diag = weights @ np.mean((U_nongripper_square[None] - nongripper_action_square) ** 2, axis=1)
        self.noise_sigma_diag = np.concatenate([gripper_sigma_diag, nongripper_sigma_diag])

        self.U += weighted_avg_noise
        # We don't return action here, because we're not actually using it.

    def sample_change_grasp(self, gripper_probabilities):
        r = self.rng.uniform(0, 1, self.n_g)
        gripper_action = r < gripper_probabilities
        return gripper_action

    def regrasp_parallel_rollout(self, phy, goal, perturbed_action, num_samples, sub_time_s, viz=None):
        gripper_probabilities = perturbed_action[:, :self.n_g]
        change_grasp = self.sample_change_grasp(gripper_probabilities)
        nongripper_action = perturbed_action[:, self.n_g:]
        nongripper_action_square = nongripper_action.reshape(num_samples, 2 * self.horizon, self.nu)
        grasp_action = nongripper_action_square[:, :self.horizon]
        move_action = nongripper_action_square[:, self.horizon:]

        # We must also copy model here because EQs are going to be changing
        args_sets = [(phy.copy_all(), self.rope_body_indices, goal, sub_time_s, *args_i) for args_i in
                     zip(change_grasp, grasp_action, move_action)]

        if viz:
            results = []
            for args in args_sets:
                result = regrasp_rollout(*args, viz)
                results.append(result)
        else:
            futures = [self.pool.submit(regrasp_rollout, *args) for args in args_sets]
            results = [f.result() for f in futures]

        # FIXME: This transformation is actually a bit slow, 50ms! But rollout time is like 15s 🤷
        grasp_results = []
        move_results = []
        grasp_change_errors = []
        for result in results:
            grasp_results.append(result[0])
            move_results.append(result[1])
            grasp_change_errors.append(result[2])

        grasp_results = list_of_tuples_to_tuple_of_lists(grasp_results)
        move_results = list_of_tuples_to_tuple_of_lists(move_results)

        grasp_results = tuple(np.array(x) for x in grasp_results)
        move_results = tuple(np.array(x) for x in move_results)
        grasp_change_errors = np.array(grasp_change_errors)

        return change_grasp, grasp_results, move_results, grasp_change_errors


def regrasp_rollout(phy, rope_body_indices, goal, sub_time_s, change_grasp, grasp_actions, move_actions, viz=None):
    if viz:
        viz.viz(phy, is_planning=True)
    grasp_results = [get_result_tuple(goal.get_results, phy, change_grasp)]

    for grasp_action in grasp_actions:
        control_step(phy, grasp_action, sub_time_s=sub_time_s)
        if viz:
            viz.viz(phy, is_planning=True)
        result_tuple = get_result_tuple(goal.get_results, phy, change_grasp)
        grasp_results.append(result_tuple)

    # Get the latest positions of the grippers
    left_tool_pos, right_tool_pos = grasp_results[-1][:2]

    # Do the grasp change
    grasp_change_error = 0
    for i, tool_pos in enumerate([left_tool_pos, right_tool_pos]):
        name = gripper_idx_to_eq_name(i)
        eq = phy.m.eq(name)
        is_grasping = bool(eq.active)
        if change_grasp[i]:
            if is_grasping:
                deactivate_eq(phy.m, name)
                settle(phy, sub_time_s, viz=viz, is_planning=True)
            else:
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
                grasp_eq = phy.m.eq(name)
                grasp_eq.obj2id = best_idx
                grasp_eq.active = 1
                grasp_eq.data[3:6] = best_offset
                grasp_change_error += best_d
    if np.any(change_grasp):
        settle(phy, sub_time_s, viz=viz, is_planning=True, settle_steps=hp['plan_settle_steps'])

    move_results = [get_result_tuple(goal.get_results, phy, change_grasp)]
    for move_action in move_actions:
        control_step(phy, move_action, sub_time_s=sub_time_s)
        if viz:
            viz.viz(phy, is_planning=True)
        result_tuple = get_result_tuple(goal.get_results, phy, change_grasp)
        move_results.append(result_tuple)

    grasp_results = list_of_tuples_to_tuple_of_lists(grasp_results)
    move_results = list_of_tuples_to_tuple_of_lists(move_results)

    return grasp_results, move_results, grasp_change_error
