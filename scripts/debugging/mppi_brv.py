import numpy as np
import time
import rerun as rr

from mjregrasping.math import softmax


def main():
    rr.init("mppi_brv")
    rr.connect()

    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    # This is me exploring what happens when you try to use MPPI with a binary action space
    # (i.e. grasp or not grasp)
    # let the cost be 1 if you don't grasp, 0 if you do grasp

    hierarchical_beta()


def hierarchical_beta():
    α = 1
    β = 1  # initial guess gives uniform distribution over p(grasp)
    temp = 0.05
    num_samples = 10
    desired_binary_state = np.array([0])
    for i in range(50):
        rr.set_time_sequence('iters', i)
        # First sample from the beta distribution to get p(grasp)
        p = np.random.beta(α, β, num_samples)

        # sample from p(grasp) to get a binary choice (p defines a binomial distribution)
        grasp = np.random.binomial(1, p).astype(float)

        # cost is a bit noisy to make the problem harder
        cost = np.square(grasp - desired_binary_state) + np.random.randn(num_samples) * 0.01

        weights = softmax(-cost, temp)

        print(f"Prior for p(grasp): α={α:.2f} β={β:.3f}")
        print(f"Weights: std={np.std(weights):.3f} max={np.max(weights):.2f}")
        rr.log_scalar('β', β)
        rr.log_tensor('p', p)
        rr.log_tensor('weights', weights)
        rr.log_tensor('grasp', grasp.astype(int))

        # Update prior
        rr.log_scalar("percent grasp", np.count_nonzero(grasp) / num_samples * 100)
        β = -1 / np.sum(weights * np.log((1 - p)))
        β = np.clip(β, 1e-9, 1e9)

        time.sleep(0.1)


def gaussian():
    # Putting a "guassian" on a probability doesn't really make sense, but seems to work in some cases
    μ = 0.1  # initial probability of grasping is 0.5
    σ = 0.05  # standard deviation of the probability of grasping
    temp = 0.1
    num_samples = 50
    for i in range(50):
        # sample noisy actions
        u = np.random.normal(μ, σ, num_samples)
        # I guess we have to clip, since probabilities must be between 0 and 1?
        u = np.clip(u, 0, 1)

        # sample from u since u is probability of grasping
        grasp = np.random.uniform(0, 1, num_samples) < u

        cost = 0 + grasp + np.random.randn(num_samples) * 0.05  # cost is a bit noisy to make the problem harder

        weights = softmax(-cost, temp)

        print(f"Prior for p(grasp): mean={μ:.2f} std={σ:.3f}")
        print(f"Noise Samples for p(grasp): mean={np.mean(u):.2f} std={np.std(u):.3f}")
        print(f"Weights: std={np.std(weights):.3f} max={np.max(weights):.2f}")
        rr.log_tensor('grasp', grasp.astype(int))
        rr.log_tensor('u', u)
        rr.log_scalar('μ', μ)
        rr.log_scalar('σ', σ)

        # Update prior
        μ = np.dot(weights, u)
        σ = np.sqrt(np.dot(weights, (μ - grasp) ** 2))

        time.sleep(0.1)


if __name__ == '__main__':
    main()
