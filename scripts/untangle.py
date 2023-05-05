#!/usr/bin/env python3
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import rospy
from mjregrasping.initialize import initialize
from mjregrasping.grasping import activate_eq
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.regrasp_mpc import RegraspMPC
from mjregrasping.rollout import control_step, DEFAULT_SUB_TIME_S


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    m, d, viz = initialize("untangle", "models/untangle_scene.xml")

    # setup_untangled_scene(m, d, mjviz)
    setup_tangled_scene(m, d, viz)

    with ThreadPoolExecutor(multiprocessing.cpu_count() - 1) as pool:
        mpc = RegraspMPC(m, pool, viz)
        mpc.run(d)


def setup_tangled_scene(m, d, viz):
    robot_q1 = np.array([
        -0.7, 0.1,  # torso
        -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
        0, 0,  # left gripper
        0.0, -0.2, 0, -0.30, 0, -0.2, 0,  # right arm
        0, 0,  # right gripper
    ])
    pid_to_joint_config(viz, m, d, robot_q1, sub_time_s=DEFAULT_SUB_TIME_S)
    robot_q2 = np.array([
        -0.5, 0.4,  # torso
        -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
        0, 0,  # left gripper
        1.2, -0.2, 0, -0.90, 0, -0.2, 0,  # right arm
        0, 0,  # right gripper
    ])
    pid_to_joint_config(viz, m, d, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)
    activate_and_settle(m, d, viz, sub_time_s=DEFAULT_SUB_TIME_S)


def setup_untangled_scene(m, d, viz):
    activate_and_settle(m, d, viz, sub_time_s=DEFAULT_SUB_TIME_S)


def activate_and_settle(m, d, viz, sub_time_s):
    # Activate the connect constraint between the rope and the gripper to
    activate_eq(m, 'right')
    # settle
    for _ in range(25):
        viz.viz(m, d)
        control_step(m, d, np.zeros(m.nu), sub_time_s=sub_time_s)
        rospy.sleep(0.01)


if __name__ == "__main__":
    main()
