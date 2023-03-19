#!/usr/bin/env python3
import multiprocessing
import concurrent
import threading
from mujoco import rollout, rollout_test
import copy
import numpy as np
from time import perf_counter
import mujoco
import argparse

N_TIME = 10
N_SUB_TIME = 20


def rollout_one_trajectory(model, data, controls):
    qs = []
    for t in range(N_TIME):
        control_t = controls[t]
        np.copyto(data.ctrl, control_t)
        for sub_t in range(N_SUB_TIME):
            mujoco.mj_step(model, data)
        qs.append(data.qpos.copy())
    return qs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xml_path", type=str)

    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)

    for n_samples  in [1, 4, 8, 80, 250, 500]:
        controls = np.zeros([N_TIME, 20])

        t0 = perf_counter()
        for sample in range(n_samples):
            data_for_rollout = copy.copy(data)
            rollout_one_trajectory(model, data_for_rollout, controls)
        dt_serial = perf_counter() - t0

        thread_local = threading.local()

        def thread_initializer():
            thread_local.data = mujoco.MjData(model)

        def call_rollout(ctrl):
            rollout_one_trajectory(model, thread_local.data, ctrl)

        chunks = [(controls,)] * n_samples

        t0 = perf_counter()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=multiprocessing.cpu_count(), initializer=thread_initializer
        ) as executor:
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(call_rollout, *chunk))
            for future in concurrent.futures.as_completed(futures):
                future.result()
        dt_parallel = perf_counter() - t0
        print(f"| {n_samples} | {dt_serial:.3f} | {dt_parallel:.3f} |")


if __name__ == "__main__":
    main()
