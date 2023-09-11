from typing import Optional
import rerun as rr

import mujoco
import numpy as np

from mjregrasping.eq_errors import compute_total_eq_error
from mjregrasping.movie import MjMovieMaker
from mjregrasping.params import hp
from mjregrasping.physics import Physics, get_full_q
from mjregrasping.real_val import RealValCommander

DEFAULT_SUB_TIME_S = 0.1


def no_results(*args, **kwargs):
    return (None,)


def control_step(phy: Physics, qvel_target, sub_time_s: float, mov: Optional[MjMovieMaker] = None, val_cmd: Optional[RealValCommander] = None):
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
        mj_q = get_full_q(phy)
        val_cmd.send_pos_command(mj_q)
        # val_cmd.pull_rope_towards_cdcpd(phy, n_sub_time)


def slow_when_eqs_bad(phy):
    speed_factor = get_speed_factor(phy)
    phy.d.ctrl *= speed_factor


def get_speed_factor(phy):
    total_eq_error = compute_total_eq_error(phy)
    speed_factor = np.clip(np.exp(-700 * total_eq_error), 0, 1)
    return speed_factor


def limit_actuator_windup(phy):
    qpos_for_act_indices = get_act_indices(phy)
    qpos_for_act = phy.d.qpos[qpos_for_act_indices]
    phy.d.act = qpos_for_act + np.clip(phy.d.act - qpos_for_act, -hp['act_windup_limit'], hp['act_windup_limit'])


def get_act_indices(phy):
    qpos_for_act_indices = phy.o.robot.qpos_indices[0] + phy.m.actuator_trnid[:, 0]
    return qpos_for_act_indices
