import numpy as np

from mjregrasping.get_result_functions import get_q_current
from mjregrasping.rollout import control_step


def pid_to_joint_config(mjviz, model, data, q_target, sub_time_s):
    kP = 5.0
    q_prev = get_q_current(model, data)
    for i in range(200):
        mjviz.viz(model, data)
        q_current = get_q_current(model, data)
        command = kP * (q_target - q_current)

        control_step(model, data, command, sub_time_s=sub_time_s)

        # get the new current q
        q_current = get_q_current(model, data)

        error = np.abs(q_current - q_target)
        max_joint_error = np.max(error)
        offending_q_idx = np.argmax(error)
        abs_qvel = np.abs(q_prev - q_current)
        offending_qvel_idx = np.argmax(abs_qvel)
        reached = np.rad2deg(max_joint_error) < 1.0
        stopped = np.rad2deg(np.max(abs_qvel)) < 1.0
        if reached and stopped:
            return

        q_prev = q_current

    if not reached:
        reason = f"Joint {model.joint(offending_q_idx).name} is {np.rad2deg(max_joint_error)} away from target."
    else:
        reason = f"Joint {model.joint(offending_qvel_idx).name} is still moving too fast."
    raise RuntimeError(f"PID failed to converge. {reason}")
