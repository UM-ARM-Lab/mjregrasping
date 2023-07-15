import mujoco
import numpy as np

from mjregrasping.eq_errors import compute_eq_errors
from mjregrasping.physics import Physics

DEFAULT_SUB_TIME_S = 0.1


def no_results(*args, **kwargs):
    return (None,)


def control_step(phy: Physics, qvel_target, sub_time_s: float):
    m = phy.m
    d = phy.d

    if qvel_target is not None:
        np.copyto(d.ctrl, qvel_target)
    else:
        print("control is None!!!")
    n_sub_time = int(sub_time_s / m.opt.timestep)

    slow_when_eqs_bad(phy)

    limit_actuator_windup(phy)

    mujoco.mj_step(m, d, nstep=n_sub_time)


def slow_when_eqs_bad(phy):
    eq_errors = compute_eq_errors(phy)
    max_eq_err = np.clip(np.max(eq_errors), 0, 1)
    speed_factor = min(max(0.0005 * -np.exp(120 * max_eq_err) + 1, 0), 1)
    phy.d.ctrl *= speed_factor


def limit_actuator_windup(phy):
    qpos_for_act_indices = phy.o.robot.qpos_indices[0] + phy.m.actuator_trnid[:, 0]
    qpos_for_act = phy.d.qpos[qpos_for_act_indices]
    phy.d.act = qpos_for_act + np.clip(phy.d.act - qpos_for_act, -0.01, 0.01)