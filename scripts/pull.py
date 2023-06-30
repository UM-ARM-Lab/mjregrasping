#!/usr/bin/env python3

import numpy as np
from transformations import quaternion_from_euler

from arc_utilities import ros_init
from mjregrasping.goals import ObjectPointGoal
from mjregrasping.grasping import activate_grasp
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.mujoco_objects import Object
from mjregrasping.regrasp_mpc_runner import Runner
from mjregrasping.rollout import DEFAULT_SUB_TIME_S


class Pull(Runner):

    def __init__(self):
        super().__init__("models/pull_scene.xml")

    def setup_scene(self, phy, viz):
        # set the rope pose
        phy.d.qpos[20:23] = np.array([1.0, 0.2, 0])
        phy.d.qpos[23:27] = quaternion_from_euler(0, 0, 0)

        robot_q2 = np.array([
            0.0, 0.9,  # torso
            1.0, 0.0, 0.0, -1.0, 0, 0, 0,  # left arm
            0.3,  # left gripper
            1.0, 0.0, 0, 0.0, 0, 0.0, 0,  # right arm
            0.1,  # right gripper
        ])
        pid_to_joint_config(phy, viz, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)

        rope = Object(phy.m, "rope")
        rope_body_indices = np.array(rope.body_indices)
        activate_grasp(phy, 'left', 0.0, rope_body_indices)
        robot_q2[9] = 0.1
        pid_to_joint_config(phy, viz, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)

    def make_goal(self, phy):
        goal_point = np.array([0.8, 0.21, 0.2])
        goal_rope_loc = 1
        goal = ObjectPointGoal(goal_point, 0.03, goal_rope_loc, self.viz)
        return goal

    def get_skeletons(self):
        return {}


@ros_init.with_ros("pull")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    runner = Pull()
    runner.run([2], "floor_obstacles")


if __name__ == "__main__":
    main()
