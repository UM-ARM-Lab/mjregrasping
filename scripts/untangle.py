#!/usr/bin/env python3
import logging
from pathlib import Path

import numpy as np

from mjregrasping.grasping import activate_eq
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.regrasp_mpc_runner import Runner
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.settle import settle

logger = logging.getLogger(f'rosout.{__name__}')


class Untangle(Runner):
    goal_point = np.array([0.78, 0.04, 1.28])
    goal_body_idx = -1
    obstacle_name = "computer_rack"
    dfield_path = Path("models/untangle_scene.dfield.pkl")
    dfield_extents = np.array([[0.6, 1.4], [-0.7, 0.4], [0.2, 1.35]])

    def __init__(self):
        super().__init__("models/untangle_scene.xml")

    def setup_scene(self, phy, viz):
        robot_q1 = np.array([
            -0.7, 0.1,  # torso
            -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
            0, 0,  # left gripper
            0.0, -0.2, 0, -0.30, 0, -0.2, 0,  # right arm
            0, 0,  # right gripper
        ])
        pid_to_joint_config(phy, viz, robot_q1, sub_time_s=DEFAULT_SUB_TIME_S)
        robot_q2 = np.array([
            -0.5, 0.4,  # torso
            -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
            0, 0,  # left gripper
            1.2, -0.2, 0, -0.90, 0, -0.6, 0,  # right arm
            0, 0,  # right gripper
        ])
        pid_to_joint_config(phy, viz, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)
        activate_eq(phy.m, 'right')
        settle(phy, sub_time_s=DEFAULT_SUB_TIME_S, viz=viz, is_planning=False)


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    runner = Untangle()
    runner.run([5])


if __name__ == "__main__":
    main()
