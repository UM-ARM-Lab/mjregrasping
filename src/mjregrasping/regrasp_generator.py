import itertools

import numpy as np

from mjregrasping.goals import ObjectPointGoal
from mjregrasping.viz import Viz


class RegraspGenerator:

    def __init__(self, op_goal: ObjectPointGoal):
        self.op_goal = op_goal
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


def get_allowable_is_grasping(n_g):
    """
    Return all possible combinations of is_grasping for n_g grippers, except for the case where no grippers are grasping
    """
    all_is_grasping = [np.array(seq) for seq in itertools.product([0, 1], repeat=n_g)]
    all_is_grasping.pop(0)
    return all_is_grasping
