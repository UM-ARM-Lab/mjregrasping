#!/usr/bin/env python3
import logging

import numpy as np
from transformations import quaternion_from_euler

from mjregrasping.body_with_children import Object
from mjregrasping.goals import CombinedGoal
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.regrasp_mpc import activate_grasp
from mjregrasping.regrasp_mpc_runner import Runner
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.settle import settle

logger = logging.getLogger(f'rosout.{__name__}')


class Untangle2(Runner):

    def __init__(self):
        super().__init__("models/pull_scene.xml")

    def setup_scene(self, phy, viz):
        # set the rope pose
        phy.d.qpos[20:23] = np.array([1.0, 0.2, 0])
        phy.d.qpos[23:27] = quaternion_from_euler(0, 0, 0)

        robot_q2 = np.array([
            0.0, 1.0,  # torso
            1.0, 0.0, 0.0, -1.0, 0, 0, 0,  # left arm
            0.3, 0.3,  # left gripper
            1.0, 0.0, 0, 0.0, 0, 0.0, 0,  # right arm
            0, 0,  # right gripper
        ])
        pid_to_joint_config(phy, viz, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)

        rope = Object(phy.m, "rope")
        rope_body_indices = np.array(rope.body_indices)
        activate_grasp(phy, 'left', 0.0, rope_body_indices)
        settle(phy, sub_time_s=DEFAULT_SUB_TIME_S, viz=viz, is_planning=False)
        print("setup")

        # initialize the grasp weights
        for i in range(11):
            self.viz.p.config[f'left_w_{i}'] = 0.0
            self.viz.p.config[f'right_w_{i}'] = 0.0
        self.viz.p.config[f'left_w_0'] = 1.0

    def make_goal(self, objects):
        goal_point = np.array([0.8, 0.21, 0.2])
        goal_body_idx = -1
        goal = CombinedGoal(goal_point, 0.05, goal_body_idx, objects, self.viz)
        return goal


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    runner = Untangle2()
    runner.run([5], "floor")  # contact cost with floor not working?


if __name__ == "__main__":
    main()
