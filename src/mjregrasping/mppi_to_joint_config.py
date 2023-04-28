import numpy as np

from mjregrasping.get_result_functions import get_q_current
from mjregrasping.rollout import control_step


def pid_to_joint_config(mjviz, model, data, q_target):
    kP = 1.0
    for t in range(100):
        mjviz.viz(model, data)
        command = kP * (q_target - get_q_current(model, data))
        control_step(model, data, command)
        # FIXME: are the joint limits wrong?
        error = np.abs(get_q_current(model, data) - q_target)
        max_joint_error = np.max(error)
        offending_joint_idx = np.argmax(error)
        if max_joint_error < np.deg2rad(1):
            return
    raise RuntimeError(f"PID failed to converge. Joint {offending_joint_idx} is {max_joint_error} away from target.")
