import logging
from typing import Optional

import numpy as np

from mjregrasping.goal_funcs import get_contact_cost
from mjregrasping.movie import MjMovieMaker
from mjregrasping.params import hp
from mjregrasping.physics import Physics, get_q
from mjregrasping.rollout import control_step, DEFAULT_SUB_TIME_S
from mjregrasping.viz import Viz


def execute_grasp_plan(phy: Physics, qs, viz: Viz, is_planning: bool, mov: Optional[MjMovieMaker] = None,
                       stop_on_contact=False):
    for q in qs[:-1]:
        pid_to_joint_config(phy, viz, q, DEFAULT_SUB_TIME_S, is_planning, mov, reached_tol=2.0, stopped_tol=10.0, stop_on_contact=stop_on_contact)
    pid_to_joint_config(phy, viz, qs[-1], DEFAULT_SUB_TIME_S, is_planning, mov, stop_on_contact=stop_on_contact)


def pid_to_joint_config(phy: Physics, viz: Viz, q_target, sub_time_s, is_planning: bool = False,
                        mov: Optional[MjMovieMaker] = None, reached_tol=1.0, stopped_tol=0.5, stop_on_contact=False):
    q_prev = get_q(phy)
    for i in range(60):
        viz.viz(phy, is_planning)
        q_current = get_q(phy)
        command = hp['joint_kp'] * (q_target - q_current)

        # copy the phy so we can check for contact before actually taking the step
        phy_plan = phy.copy_data()
        control_step(phy_plan, command, sub_time_s=sub_time_s)
        plan_contact_cost = get_contact_cost(phy_plan)
        if plan_contact_cost > 0.0 and stop_on_contact:
            print(f"Contact detected, stopping!")
            return

        # take the step on the real phy
        control_step(phy, command, sub_time_s=sub_time_s, mov=mov)

        # get the new current q
        q_current = get_q(phy)

        error = np.abs(q_current - q_target)
        max_joint_error = np.max(error)
        offending_q_idx = np.argmax(error)
        abs_qvel = np.abs(q_prev - q_current)
        offending_qvel_idx = np.argmax(abs_qvel)
        # NOTE: this assumes all joints are rotational...
        reached = np.rad2deg(max_joint_error) < reached_tol
        stopped = np.rad2deg(np.max(abs_qvel)) < stopped_tol
        if reached and stopped:
            return

        q_prev = q_current

    if not reached:
        reason = f"qpos {offending_q_idx} is {np.rad2deg(max_joint_error)} deg away from target."
    else:
        reason = f"qpos {offending_qvel_idx} is still moving too fast."
    print(f"PID failed to converge. {reason}")
