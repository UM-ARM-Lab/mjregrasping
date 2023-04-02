import concurrent
import copy
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

MAX_VEL_TILL_ERROR_RAD = np.deg2rad(3)

N_SUB_TIME_S = 0.1


def rollout(model, data, controls, get_result_func=None):
    joint_indices_for_actuators = model.actuator_trnid[:, 0]
    results_lists = None
    if get_result_func is not None:
        result_tuple = get_result_tuple(data, get_result_func, model)

        if results_lists is None:
            results_lists = tuple([] for _ in result_tuple)

        for result_list, result in zip(results_lists, result_tuple):
            result_list.append(result)

    for t in range(controls.shape[0]):
        qvel_target = controls[t]
        # q_target = controls[t]
        #
        # q_current = data.qpos[joint_indices_for_actuators]
        # qvel_current = data.qvel[joint_indices_for_actuators]
        #
        # qerr = q_target - q_current
        #
        # # do the lowest-level control here
        # v_max = model.actuator_ctrlrange[:, 1]  # NOTE: assumes symmetric range
        # kv = v_max / MAX_VEL_TILL_ERROR_RAD
        # qvel_target = qerr * kv
        #
        # qvel_err = qvel_target - qvel_current
        # data.userdata = np.concatenate([qerr, qvel_target, qvel_err])

        control_step(model, data, qvel_target)

        if get_result_func is not None:
            result_tuple = get_result_tuple(data, get_result_func, model)

            if results_lists is None:
                results_lists = tuple([] for _ in result_tuple)

            for result_list, result in zip(results_lists, result_tuple):
                result_list.append(result)

    if len(results_lists) == 1:
        return results_lists[0]
    return results_lists


def get_result_tuple(data, get_result_func, model):
    result_tuple = get_result_func(model, data)
    if not isinstance(result_tuple, tuple):
        result_tuple = (result_tuple,)

    # make sure we copy otherwise the data gets overwritten
    result_tuple = tuple(np.copy(result) for result in result_tuple)

    return result_tuple


def control_step(model, data, qvel_target):
    if qvel_target is not None:
        np.copyto(data.ctrl, qvel_target)
    else:
        print("control is None!!!")
    n_sub_time = int(N_SUB_TIME_S / model.opt.timestep)
    mujoco.mj_step(model, data, nstep=n_sub_time)


def parallel_rollout(pool: ThreadPoolExecutor, model, data, controls_samples, get_result_func=None):
    args_sets = [(model, copy.copy(data), controls, get_result_func) for controls in controls_samples]
    futures = [pool.submit(rollout, *args) for args in args_sets]

    results = [f.result() for f in futures]

    result0 = results[0]
    if isinstance(result0, tuple):
        results = tuple(np.array(result_i) for result_i in zip(*results))
    else:
        results = np.array(results)

    return results
