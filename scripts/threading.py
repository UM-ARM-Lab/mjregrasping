#!/usr/bin/env python3
import logging

import mujoco
import numpy as np
from transformations import quaternion_from_matrix, quaternion_from_euler

from mjregrasping.body_with_children import Object
from mjregrasping.goals import CombinedThreadingGoal
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.regrasp_mpc import activate_weld
from mjregrasping.regrasp_mpc_runner import Runner
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.settle import settle

logger = logging.getLogger(f'rosout.{__name__}')


class Threading1(Runner):

    def __init__(self):
        super().__init__("models/threading_scene.xml")

    def setup_scene(self, phy, viz):
        robot_q2 = np.array([
            0.1, 1.05,  # torso
            1.0, 0.0, 0.0, -1.0, 1.5707, -0.6, 1.5707,  # left arm
            0.3,  # left gripper
            -.5, 0.0, 0, 0.5, 0, 0.0, 0,  # right arm
            0,  # right gripper
        ])
        pid_to_joint_config(phy, viz, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)

        rope = Object(phy.m, "rope")
        rope_body_indices = np.array(rope.body_indices)

        loc = 0.08
        name = 'left'
        activate_weld(phy, name, loc, rope_body_indices)
        settle(phy, sub_time_s=DEFAULT_SUB_TIME_S, viz=viz, is_planning=False)

        robot_q2 = np.array([
            0.1, 0.0,  # torso
            1.0, 0.0, 0.0, 0, 1.5707, 0, 1.5707,  # left arm
            0.3,  # left gripper
            -.5, 0.0, 0, 0.5, 0, 0.0, 0,  # right arm
            0,  # right gripper
        ])
        pid_to_joint_config(phy, viz, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)
        print("setup")


    def make_goal(self, objects):
        goal_point = np.array([0.8, 0.0, 0.0])
        goal_body_idx = 0
        goal = CombinedThreadingGoal(goal_point, 0.05, goal_body_idx, objects, self.viz)
        return goal


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    runner = Threading1()
    runner.run([0], obstacle_name="cable_tube")


if __name__ == "__main__":
    main()
