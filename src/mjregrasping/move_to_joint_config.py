from typing import Optional

import numpy as np

from mjregrasping.movie import MjMovieMaker
from mjregrasping.params import hp
from mjregrasping.physics import Physics, get_q
from mjregrasping.real_val import RealValCommander
from mjregrasping.rollout import control_step, DEFAULT_SUB_TIME_S
from mjregrasping.val_dup import val_dedup
from mjregrasping.viz import Viz
from moveit_msgs.msg import MotionPlanResponse


def pid_to_joint_configs(phy: Physics, res: MotionPlanResponse, viz: Viz, is_planning: bool,
                         mov: Optional[MjMovieMaker] = None, val_cmd: Optional[RealValCommander] = None, reached_tol=3.0):
    qs = np.array([p.positions for p in res.trajectory.joint_trajectory.points])
    for q in qs[:-1]:
        q_current = get_q(phy)
        error = np.abs(q_current - q)
        max_joint_error = np.max(error)
        reached = np.rad2deg(max_joint_error) < reached_tol
        if reached:
            continue
        pid_to_joint_config(phy, viz, q, DEFAULT_SUB_TIME_S, is_planning, mov, val_cmd, reached_tol=reached_tol, stopped_tol=10)
    pid_to_joint_config(phy, viz, qs[-1], DEFAULT_SUB_TIME_S, is_planning, mov, val_cmd)
    if val_cmd:
        while True:
            pid_to_joint_config(phy, viz, qs[-1], DEFAULT_SUB_TIME_S, is_planning, mov, val_cmd)
            current_q = val_dedup(val_cmd.get_latest_qpos_in_mj_order())
            error = np.abs(current_q - qs[-1])
            # ignore grippers
            error[9] = 0
            error[17] = 0
            max_joint_error = np.rad2deg(np.max(error))
            reached = max_joint_error < reached_tol
            if reached:
                break


def warn_about_limits(q_target, phy):
    low = phy.m.actuator_actrange[:, 0]
    high = phy.m.actuator_actrange[:, 1]
    if np.any(q_target > high + 0.03):
        offending_idx = np.argmin(high - q_target)
        name = phy.m.actuator(offending_idx).name
        print(f"q_target {q_target[offending_idx]} is above actuator limit {high[offending_idx]} for joint {name}!")
    if np.any(q_target < low - 0.03):
        offending_idx = np.argmin(q_target - low)
        name = phy.m.actuator(offending_idx).name
        print(f"q_target {q_target[offending_idx]} is below actuator limit {low[offending_idx]} for joint {name}!")


def pid_to_joint_config(phy: Physics, viz: Optional[Viz], q_target, sub_time_s, is_planning: bool = False,
                        mov: Optional[MjMovieMaker] = None, val_cmd: Optional[RealValCommander] = None, reached_tol=1.0, stopped_tol=0.5):
    q_prev = get_q(phy)
    for i in range(75):
        if viz:
            viz.viz(phy, is_planning)
        q_current = get_q(phy)
        command = hp['joint_kp'] * (q_target - q_current)

        # take the step on the real phy
        control_step(phy, command, sub_time_s=sub_time_s, mov=mov, val_cmd=val_cmd)

        # get the new current q
        q_current = get_q(phy)

        warn_about_limits(q_target, phy)

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
        elif stopped and i > 10:
            break

        q_prev = q_current

    if not reached:
        name = phy.m.actuator(offending_q_idx).name
        reason = f"actuator {name} is {np.rad2deg(max_joint_error)} deg away from target."
    else:
        reason = f"qpos {offending_qvel_idx} is still moving too fast."
    print(f"PID failed to converge. {reason}")
