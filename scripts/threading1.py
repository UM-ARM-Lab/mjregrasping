#!/usr/bin/env python3
import logging

import numpy as np
from transformations import quaternion_from_euler

from mjregrasping.body_with_children import Object
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.regrasp_mpc import activate_grasp
from mjregrasping.regrasp_mpc_runner import Runner
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.settle import settle

logger = logging.getLogger(f'rosout.{__name__}')


class Threading1(Runner):

    def __init__(self):
        super().__init__("models/threading_scene1.xml")

    def setup_scene(self, phy, viz):
        # set the rope pose
        phy.d.qpos[20:23] = np.array([1.0, 0.2, 0])
        phy.d.qpos[23:27] = quaternion_from_euler(0, 0, 1.5707)

        robot_q2 = np.array([
            0.0, 1.0,  # torso
            1.0, 0.0, 0.0, -1.0, 0, 0, 0,  # left arm
            0.3, 0.3,  # left gripper
            -.5, 0.0, 0, 0.5, 0, 0.0, 0,  # right arm
            0, 0,  # right gripper
        ])
        pid_to_joint_config(phy, viz, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)

        rope = Object(phy.m, "rope")
        rope_body_indices = np.array(rope.body_indices)
        activate_grasp(phy, 'left', 0.0, rope_body_indices)
        settle(phy, sub_time_s=DEFAULT_SUB_TIME_S, viz=viz, is_planning=False)
        print("setup")

    def make_goal(self, objects):
        goal = CombinedThreadingGoal(goal_point, 0.05, goal_body_idx, objects, self.viz)
        return goal


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    runner = Threading1()
    runner.run([0], obstacle_name="cable_tube")


if __name__ == "__main__":
    main()
