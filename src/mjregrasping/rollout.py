from typing import Optional

import mujoco
import numpy as np

from mjregrasping.eq_errors import compute_eq_errors
from mjregrasping.movie import MjMovieMaker
from mjregrasping.params import hp
from mjregrasping.physics import Physics, get_full_q, get_qpos_for_actuators
from mjregrasping.real_val import RealValCommander

DEFAULT_SUB_TIME_S = 0.1


def no_results(*args, **kwargs):
    return (None,)


def control_step(phy: Physics, qvel_target, sub_time_s: float, mov: Optional[MjMovieMaker] = None,
                 val_cmd: Optional[RealValCommander] = None):
    m = phy.m
    d = phy.d

    if qvel_target is not None:
        np.copyto(d.ctrl, qvel_target)
    else:
        print("control is None!!!")
    n_sub_time = int(sub_time_s / m.opt.timestep)

    slow_when_eqs_bad(phy)

    limit_actuator_windup(phy)

    if mov:
        # This renders every frame
        for _ in range(n_sub_time):
            mujoco.mj_step(m, d, nstep=1)
            mov.render(d)
    else:
        mujoco.mj_step(m, d, nstep=n_sub_time)

    if val_cmd:
        val_cmd.send_pos_command(get_full_q(phy))
        val_cmd.pull_rope_towards_cdcpd(phy, n_sub_time)


def slow_when_eqs_bad(phy):
    eq_errors = compute_eq_errors(phy)
    max_eq_err = np.clip(np.max(eq_errors), 0, 1)
    speed_factor = min(max(0.001 * -np.exp(150 * max_eq_err) + 1, 0), 1)
    phy.d.ctrl *= speed_factor


def limit_actuator_windup(phy):
    qpos_for_act = get_qpos_for_actuators(phy)
    phy.d.act = qpos_for_act + np.clip(phy.d.act - qpos_for_act, -hp['act_windup_limit'], hp['act_windup_limit'])
