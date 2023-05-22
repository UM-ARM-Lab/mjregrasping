import logging

import numpy as np

from mjregrasping.rollout import parallel_rollout

logger = logging.getLogger(f'rosout.{__name__}')


def softmax(x, temp):
    x = x / temp
    return np.exp(x) / np.exp(x).sum()


class MujocoMPPI:

    def __init__(self, pool, nu, seed, horizon, noise_sigma, lambda_=1., gamma=0.9):
        self.pool = pool
        self.horizon = horizon
        self.nu = nu
        self.gamma = gamma
        self.seed = seed

        # dimensions of state and control
        self.lambda_ = lambda_

        self.noise_sigma = np.eye(horizon * nu) * noise_sigma
        self.noise_rng = np.random.RandomState(seed)

        self.U = None
        self.reset()

        # sampled results from last command
        self.cost = None
        self.costs = None
        self.cost_normalized = None
        self.rollout_results = None
        self.actions = None

    def roll(self):
        """ shift command 1 time step. Used before sampling a new command. """
        self.U[:-self.nu] = self.U[self.nu:]
        # sample a new random reference control for the last time step
        self.U[-self.nu:] = (self.noise_rng.randn(self.horizon * self.nu) @ self.noise_sigma)[-self.nu:]

    def roll_and_command(self, phy, get_result_func, cost_func, sub_time_s, num_samples):
        """
        Use this for no warmstarting.

        cost func needs to take in the output of get_result_func and return a cost for each sample.
        get_result_func needs to take in the model and data and return a result for each sample, which
        can be any object or tuple of objects.
        """
        self.roll()

        return self.command(phy, get_result_func, cost_func, sub_time_s, num_samples)

    def command(self, phy, get_result_func, cost_func, sub_time_s, num_samples):
        """
        cost func needs to take in the output of get_result_func and return a cost for each sample.
        get_result_func needs to take in the model and data and return a result for each sample, which
        can be any object or tuple of objects.
        """
        noise = self.noise_rng.randn(num_samples, self.horizon * self.nu) @ self.noise_sigma
        perturbed_action = self.U + noise

        lower = np.repeat(phy.m.actuator_ctrlrange[:, 0], self.horizon)
        upper = np.repeat(phy.m.actuator_ctrlrange[:, 1], self.horizon)

        # NOTE: We clip the absolute action, then recompute the noise
        #  so that later when we weight noise, we're weighting the bounded noise.
        perturbed_action = np.clip(perturbed_action, lower, upper)
        noise = perturbed_action - self.U

        perturbed_action_square = perturbed_action.reshape(num_samples, self.horizon, self.nu)
        results = parallel_rollout(self.pool, phy, perturbed_action_square, sub_time_s, get_result_func)

        self.rollout_results = results
        self.actions = perturbed_action_square
        self.costs = cost_func(results)

        gammas = np.power(self.gamma, np.arange(self.horizon))[None]
        self.cost = np.sum(gammas * self.costs, axis=-1)

        # normalized cost is only used for visualization, so we avoid numerical issues
        cost_range = (self.cost.max() - self.cost.min())
        if cost_range < 1e-6:
            cost_range = 1.0
        self.cost_normalized = (self.cost - self.cost.min()) / cost_range

        weights = softmax(-self.cost_normalized, self.lambda_)
        logger.debug(f'weights: std={float(np.std(weights)):.2f} max={float(np.max(weights)):.2f}')

        # compute the (weighted) average noise and add that to the reference control
        weighted_avg_noise = np.sum(weights[:, None] * noise, axis=0)

        # Covariance matrix adaptation? Not working currently...
        # new_noise_sigma = np.eye(self.noise_sigma.shape[0])
        # for k in range(num_samples):
        #     theta_k = perturbed_action[k]
        #     theta = self.U
        #     p_k = weights[k]
        #     new_sigma_k = p_k * np.outer(theta_k - theta, theta_k - theta)
        #     new_noise_sigma += new_sigma_k
        # self.noise_sigma = new_noise_sigma

        self.U += weighted_avg_noise
        action = self.U[:self.nu]

        return action

    def reset(self):
        """
        Resets the control samples.
        """
        self.U = np.zeros([self.horizon * self.nu])
        # resetting the RNG makes things a bit more deterministic, so easier for debugging
        self.noise_rng = np.random.RandomState(self.seed)

    def get_min_terminal_cost(self):
        """ Returns the minimum cost of the last time step """
        return self.costs[:, -1].min()

    def get_first_step_cost(self):
        """ Returns the minimum cost of the first time step """
        return self.costs[:, 0].min()
