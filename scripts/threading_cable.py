#!/usr/bin/env python3
import logging

import numpy as np

from mjregrasping.goals import CombinedThreadingGoal
from mjregrasping.regrasp_mpc_runner import Runner
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.settle import settle

logger = logging.getLogger(f'rosout.{__name__}')


class Threading1(Runner):

    def __init__(self):
        super().__init__("models/threading_scene.xml")

    def setup_scene(self, phy, viz):
        settle(phy, sub_time_s=DEFAULT_SUB_TIME_S, viz=viz, is_planning=False)

        # loc = 0.08
        # name = 'left'
        # activate_weld(phy, name, loc, rope_body_indices)
        # settle(phy, sub_time_s=DEFAULT_SUB_TIME_S, viz=viz, is_planning=False)

    def make_goal(self, phy, objects):
        goal_point = phy.d.site("cable_tube_center").xpos
        goal_dir = np.array([0.0, -1.0, 0.0])
        goal_body_idx = 0
        goal = CombinedThreadingGoal(goal_point, goal_dir, 0.05, goal_body_idx, objects, self.viz)
        return goal


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    runner = Threading1()
    runner.run([0], obstacle_name="cable_tube")


if __name__ == "__main__":
    main()
