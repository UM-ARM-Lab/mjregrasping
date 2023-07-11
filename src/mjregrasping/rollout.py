import mujoco
import numpy as np

from mjregrasping.eq_errors import compute_eq_errors
from mjregrasping.physics import Physics

DEFAULT_SUB_TIME_S = 0.1


def no_results(*args, **kwargs):
    return (None,)


def rollout(phy, controls, sub_time_s, get_result_func=no_results, get_result_args=None):
    if get_result_args is None:
        get_result_args = []
    # run for the initial data, so that the current state is returned in the output
    results = [get_result_func(phy, *get_result_args)]

    for t in range(controls.shape[0]):
        qvel_target = controls[t]

        control_step(phy, qvel_target, sub_time_s=sub_time_s)

        results_t = get_result_func(phy, *get_result_args)

        results.append(results_t)

    results = np.stack(results, dtype=object, axis=1)
    return results


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
    qpos_for_act = phy.d.qpos[phy.m.actuator_trnid[:, 0]]
    phy.d.act = qpos_for_act + np.clip(phy.d.act - qpos_for_act, -0.01, 0.01)


def parallel_rollout(pool, phy, controls_samples, sub_time_s: float, get_result_func=no_results):
    # within a rollout you're not changing the _model_, so we use copy_data since it's faster
    args_sets = [(phy.copy_data(), controls, sub_time_s, get_result_func) for controls in controls_samples]
    futures = [pool.submit(rollout, *args) for args in args_sets]
    results = np.stack([f.result() for f in futures], dtype=object, axis=1)
    return results


def expand_result(result):
    results = []
    for r in result:
        if isinstance(r, np.ndarray):
            results.append(r[None])
        else:
            results.append([r])

    return tuple(results)
