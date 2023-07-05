import numpy as np

from mjregrasping.viz import Viz


class RegraspGenerator:

    def __init__(self, op_goal, viz: Viz):
        self.op_goal = op_goal
        self.viz = viz
        self.rng = np.random.RandomState(0)

    def generate(self, phy):
        """
        Args:
            phy: Current state of the world

        Returns:
            the next grasp locations. This is a vector of size [n_g] where the element is -1 if there is no grasp,
            and between 0 and 1 if there is a grasp. A grasp of 0 means one end, 0.5 means the middle, and 1 means
            the other end.
        """
        raise NotImplementedError()
