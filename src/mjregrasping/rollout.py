import concurrent
import copy
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

N_SUB_TIME = 20


def rollout(model, data, controls):
    qs = []
    for t in range(controls.shape[0]):
        control_t = controls[t]
        np.copyto(data.ctrl, control_t)
        for sub_t in range(N_SUB_TIME):
            mujoco.mj_step(model, data)
        qs.append(data.qpos.copy())
    return qs


def parallel_rollout(model, data, controls_samples):
    thread_local = threading.local()

    def init():
        thread_local.data = copy.copy(data)

    def _rollout(ctrl):
        return rollout(model, thread_local.data, ctrl)

    with ThreadPoolExecutor(multiprocessing.cpu_count(), initializer=init) as pool:
        futures = [pool.submit(_rollout, controls) for controls in controls_samples]

    results = np.array([f.result() for f in concurrent.futures.as_completed(futures)])

    return results
