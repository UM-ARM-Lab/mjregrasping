from typing import Optional
import rerun as rr

import mujoco
import numpy as np

from mjregrasping.eq_errors import compute_total_eq_error
from mjregrasping.movie import MjMovieMaker
from mjregrasping.params import hp
from mjregrasping.physics import Physics, get_full_q, get_qpos_for_actuators
from mjregrasping.real_val import RealValCommander
from dm_control.mujoco.wrapper.mjbindings import mjlib

DEFAULT_SUB_TIME_S = 0.1

USEFUL_INDICES_vel = [81, 82, 83, 84, 85, 86, 87, 88, 89, 92, 93, 94, 95, 96, 97, 98]
USEFUL_INDICES_pos = [107, 108, 109, 110, 111, 112, 113, 114, 115, 118, 119, 120, 121, 122, 123, 124]
def velocity_control(gripper_delta, physics, n_sub_time):
    jac_pos_l = np.zeros((3, physics.model.nv))
    jac_rot_l = np.zeros((3, physics.model.nv))
    jac_pos_r = np.zeros((3, physics.model.nv))
    jac_rot_r = np.zeros((3, physics.model.nv))

    mjlib.mj_jacGeom(physics.model.ptr, physics.data.ptr, jac_pos_l, jac_rot_l, physics.model.name2id('untangle/left_finger_pad', 'geom'))
    mjlib.mj_jacGeom(physics.model.ptr, physics.data.ptr, jac_pos_r, jac_rot_r, physics.model.name2id('untangle/right_finger_pad', 'geom'))

    J = np.concatenate((jac_pos_l[:, USEFUL_INDICES_vel], jac_pos_r[:, USEFUL_INDICES_vel]), axis=0)
    J_T = J.T
    ctrl = J_T @ np.linalg.solve(J @ J_T + 1e-6 * np.eye(6), gripper_delta)
    current_qpos = physics.data.qpos[USEFUL_INDICES_pos]

    vmin = physics.model.actuator_ctrlrange[:, 0]
    vmax = physics.model.actuator_ctrlrange[:, 1]
    #Create a list of length 10 that interpolates between the current position and the desired position
    frac = np.linspace(1/n_sub_time, 1, n_sub_time)
    qpos_list = [np.clip(current_qpos + frac[i] * ctrl, vmin, vmax) for i in range(len(frac))]

    return qpos_list

def no_results(*args, **kwargs):
    return (None,)


def control_step(phy: Physics, eef_delta_target, sub_time_s: float, mov: Optional[MjMovieMaker] = None,
                 val_cmd: Optional[RealValCommander] = None):
    m = phy.m
    d = phy.d

    n_sub_time = int(sub_time_s / m.opt.timestep)

    if eef_delta_target is not None:
        setpoints = velocity_control(eef_delta_target, phy.p, n_sub_time)
    else:
        print("control is None!!!")

    # slow_when_eqs_bad(phy)

    # limit_actuator_windup(phy)

    if mov:
        # This renders every frame
        for _ in range(n_sub_time):
            mujoco.mj_step(m, d, nstep=1)
            mov.render(d)
    else:
        for i in range(n_sub_time):
            phy.p.set_control(setpoints[i])
            phy.p.data.qpos[USEFUL_INDICES_pos] = setpoints[i]
            phy.p.step()
    if val_cmd:
        mj_q = get_full_q(phy)
        val_cmd.send_pos_command(mj_q, slow=slow)
        # val_cmd.pull_rope_towards_cdcpd(phy, n_sub_time / 4)


def slow_when_eqs_bad(phy):
    speed_factor = get_speed_factor(phy)
    phy.d.ctrl *= speed_factor


def get_speed_factor(phy):
    return 1
    total_eq_error = compute_total_eq_error(phy)
    speed_factor = np.clip(np.exp(-700 * total_eq_error), 0, 1)
    return speed_factor


def limit_actuator_windup(phy):
    qpos_for_act = get_qpos_for_actuators(phy)
    phy.d.act = qpos_for_act + np.clip(phy.d.act - qpos_for_act, -hp['act_windup_limit'], hp['act_windup_limit'])
