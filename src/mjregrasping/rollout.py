import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List

import mujoco
import numpy as np

from mjregrasping.grasping import compute_eq_errors
from mjregrasping.physics import Physics

MAX_VEL_TILL_ERROR_RAD = np.deg2rad(3)
DEFAULT_SUB_TIME_S = 0.1

logger = logging.getLogger(f'rosout.{__name__}')


def no_results(*args, **kwargs):
    return (None,)


# noinspection PyTypeChecker
def list_of_tuples_to_tuple_of_arrays(results: List[Tuple]) -> Tuple[np.array]:
    return tuple(np.array(result) for result in zip(*results))


def rollout(phy, controls, sub_time_s, get_result_func=no_results, get_result_args=None):
    if get_result_args is None:
        get_result_args = []
    # run for the initial data, so that the current state is returned in the output
    results: List[Tuple] = [get_result_tuple(get_result_func, phy, *get_result_args)]

    for t in range(controls.shape[0]):
        qvel_target = controls[t]

        control_step(phy, qvel_target, sub_time_s=sub_time_s)

        result_tuple: Tuple = get_result_tuple(get_result_func, phy, *get_result_args)

        results.append(result_tuple)

    return list_of_tuples_to_tuple_of_arrays(results)


def get_result_tuple(get_result_func, phy, *get_result_args):
    result_tuple = get_result_func(phy, *get_result_args)

    # make sure we copy otherwise the data gets overwritten
    result_tuple = tuple(np.copy(result) for result in result_tuple)

    return result_tuple


def control_step(phy: Physics, qvel_target, sub_time_s: float):
    m = phy.m
    d = phy.d

    if qvel_target is not None:
        np.copyto(d.ctrl, qvel_target)
    else:
        logger.warning("control is None!!!")
    n_sub_time = int(sub_time_s / m.opt.timestep)

    slow_when_eqs_bad(phy)

    limit_actuator_windup(phy)

    mujoco.mj_step(m, d, nstep=n_sub_time)


def slow_when_eqs_bad(phy):
    eq_errors = compute_eq_errors(phy)
    max_eq_err = np.max(eq_errors)
    speed_factor = min(max(0.0005 * -np.exp(120 * max_eq_err) + 1, 0), 1)
    phy.d.ctrl *= speed_factor


# Very small performance optimization to avoid re-computing these. We assume it's constant.
_qpos_indices_for_act = np.array([
    0, 1,
    2, 3, 4, 5, 6, 7, 8,
    9,
    11, 12, 13, 14, 15, 16, 17,
    18,
])


def limit_actuator_windup(phy):
    qpos_for_act = phy.d.qpos[_qpos_indices_for_act]
    phy.d.act = qpos_for_act + np.clip(phy.d.act - qpos_for_act, -0.01, 0.01)


def parallel_rollout(pool, phy, controls_samples, sub_time_s: float, get_result_func=no_results):
    # within a rollout you're not changing the _model_, so we use copy_data since it's faster
    args_sets = [(phy.copy_data(), controls, sub_time_s, get_result_func) for controls in controls_samples]
    futures = [pool.submit(rollout, *args) for args in args_sets]
    results = [f.result() for f in futures]
    results = tuple(np.array(result_i) for result_i in zip(*results))
    return results


def expand_result(result):
    results = []
    for r in result:
        if isinstance(r, np.ndarray):
            results.append(r[None])
        else:
            results.append([r])

    return tuple(results)
