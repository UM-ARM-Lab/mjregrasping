import logging
from copy import copy

import numpy as np

from mjregrasping.physics import Physics
from mjregrasping.rollout import control_step
from mjregrasping.viz import Viz

logger = logging.getLogger(f'rosout.{__name__}')


def pid_to_joint_config(phy: Physics, viz: Viz, q_target, sub_time_s):
    kP = 5.0
    q_prev = get_q_current(phy)
    for i in range(100):
        viz.viz(phy)
        q_current = get_q_current(phy)
        command = kP * (q_target - q_current)

        control_step(phy, command, sub_time_s=sub_time_s)

        # get the new current q
        q_current = get_q_current(phy)

        error = np.abs(q_current - q_target)
        max_joint_error = np.max(error)
        offending_q_idx = np.argmax(error)
        abs_qvel = np.abs(q_prev - q_current)
        offending_qvel_idx = np.argmax(abs_qvel)
        reached = np.rad2deg(max_joint_error) < 1
        stopped = np.rad2deg(np.max(abs_qvel)) < 0.5
        if reached and stopped:
            return

        q_prev = q_current

    if not reached:
        reason = f"qpos {offending_q_idx} is {np.rad2deg(max_joint_error)} deg away from target."
    else:
        reason = f"qpos {offending_qvel_idx} is still moving too fast."
    # raise RuntimeError()
    logger.error(f"PID failed to converge. {reason}")


def get_q_current(phy):
    # NOTE: if I used "sensors" instead of "qpos", it might clearer because I could just omit the sensor
    #  for the mimic'd gripper joint.
    deduplicated_indices = np.array([0, 1,
                                     2, 3, 4, 5, 6, 7, 8,
                                     9,
                                     11, 12, 13, 14, 15, 16, 17,
                                     18])
    return copy(phy.d.qpos[deduplicated_indices])
