#!/usr/bin/env python3
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import rospy
from mjregrasping.initialize import initialize, activate_eq
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.regrasp_mpc import RegraspMPC
from mjregrasping.rollout import control_step, DEFAULT_SUB_TIME_S


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    model, data, mjviz, viz_pubs = initialize("untangle", "models/untangle_scene.xml")

    # setup_untangled_scene(model, data, mjviz)
    setup_tangled_scene(model, data, mjviz)

    with ThreadPoolExecutor(multiprocessing.cpu_count() - 1) as pool:
        mpc = RegraspMPC(model, mjviz, viz_pubs, pool)
        mpc.run(data)


def setup_tangled_scene(model, data, mjviz):
    robot_q1 = np.array([
        -0.7, 0.1,  # torso
        -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
        0, 0,  # left gripper
        0.0, -0.2, 0, -0.30, 0, -0.2, 0,  # right arm
        0, 0,  # right gripper
    ])
    pid_to_joint_config(mjviz, model, data, robot_q1, sub_time_s=DEFAULT_SUB_TIME_S)
    robot_q2 = np.array([
        -0.5, 0.4,  # torso
        -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
        0, 0,  # left gripper
        1.2, -0.2, 0, -0.90, 0, -0.2, 0,  # right arm
        0, 0,  # right gripper
    ])
    pid_to_joint_config(mjviz, model, data, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)
    activate_and_settle(data, mjviz, model, sub_time_s=DEFAULT_SUB_TIME_S)


def setup_untangled_scene(model, data, mjviz):
    activate_and_settle(data, mjviz, model)


def activate_and_settle(data, mjviz, model, sub_time_s):
    # Activate the connect constraint between the rope and the gripper to
    activate_eq(model, 'right')
    # settle
    for _ in range(25):
        mjviz.viz(model, data)
        control_step(model, data, np.zeros(model.nu), sub_time_s=sub_time_s)
        rospy.sleep(0.01)


if __name__ == "__main__":
    main()
