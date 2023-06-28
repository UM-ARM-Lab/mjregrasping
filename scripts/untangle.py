#!/usr/bin/env python3
import logging

import numpy as np

from mjregrasping.magnetic_fields import load_skeletons
from mjregrasping.mujoco_objects import Object
from mjregrasping.goals import ObjectPointGoal
from mjregrasping.grasping import activate_grasp
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.regrasp_mpc_runner import Runner
from mjregrasping.rollout import DEFAULT_SUB_TIME_S

logger = logging.getLogger(f'rosout.{__name__}')



class Untangle(Runner):

    def __init__(self):
        super().__init__("models/untangle_scene.xml")

    def setup_scene(self, phy, viz):
        robot_q1 = np.array([
            -0.7, 0.1,  # torso
            -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
            0,  # left gripper
            0.0, -0.2, 0, -0.30, 0, -0.2, 0,  # right arm
            0.3,  # right gripper
        ])
        pid_to_joint_config(phy, viz, robot_q1, sub_time_s=DEFAULT_SUB_TIME_S)
        robot_q2 = np.array([
            -0.5, 0.4,  # torso
            -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
            0.1,  # left gripper
            1.2, -0.2, 0, -0.90, 0, -0.6, 0,  # right arm
            0.3,  # right gripper
        ])
        pid_to_joint_config(phy, viz, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)
        rope = Object(phy.m, "rope")
        rope_body_indices = np.array(rope.body_indices)
        activate_grasp(phy, 'right', 0.95, rope_body_indices)
        robot_q2[-1] = 0.05  # close right gripper
        pid_to_joint_config(phy, viz, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)

    def make_goal(self, phy):
        goal_point = np.array([0.78, 0.04, 1.25])
        loc = 1
        # goal = CombinedGoal(goal_point, 0.05, goal_body_idx, self.viz)
        goal = ObjectPointGoal(goal_point, 0.05, loc, self.viz)
        return goal

    def get_skeletons(self):
        return load_skeletons("models/computer_rack_skeleton.hjson")

    def get_attach_pos(self, phy):
        return phy.d.body("attach").xpos

def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    runner = Untangle()
    runner.run([5], obstacle_name="computer_rack")


if __name__ == "__main__":
    main()
