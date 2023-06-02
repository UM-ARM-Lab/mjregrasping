from typing import Optional, Callable

import numpy as np

from mjregrasping.grasping import activate_eq
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.movie import MjMovieMaker
from mjregrasping.physics import Physics
from mjregrasping.rollout import DEFAULT_SUB_TIME_S, control_step, get_result_tuple
from mjregrasping.viz import Viz


def setup_tangled_scene(phy: Physics, viz):
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
    activate_and_settle(phy, viz, sub_time_s=DEFAULT_SUB_TIME_S)


def setup_untangled_scene(phy, viz):
    activate_and_settle(phy, viz, sub_time_s=DEFAULT_SUB_TIME_S)


def activate_and_settle(phy, viz, sub_time_s, is_planning=False):
    # Activate the connect constraint between the rope and the gripper to
    activate_eq(phy.m, 'right')
    # settle
    settle(phy, sub_time_s, viz, is_planning)


def settle(phy, sub_time_s, viz: Viz, is_planning, settle_steps=50, mov: Optional[MjMovieMaker] = None,
           ctrl: Optional[np.ndarray] = None, get_result_func: Optional[Callable] = None):
    if ctrl is None:
        ctrl = np.zeros(phy.m.nu)

    results_lists = None
    for _ in range(settle_steps):
        viz.viz(phy, is_planning)
        control_step(phy, ctrl, sub_time_s=sub_time_s)

        if get_result_func is not None:
            result_tuple = get_result_tuple(get_result_func, phy)

            if results_lists is None:
                results_lists = tuple([] for _ in result_tuple)

            for result_list, result in zip(results_lists, result_tuple):
                result_list.append(result)

        if not is_planning and mov:
            mov.render(phy.d)

    if results_lists is None:
        return None

    return tuple(np.array(result_i) for result_i in results_lists)
