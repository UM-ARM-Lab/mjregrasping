#!/usr/bin/env python3
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from time import perf_counter
import mujoco
import argparse
from mjregrasping.rollout import parallel_rollout

N_TIME = 10
N_SUB_TIME = 50



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xml_path", type=str)

    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)

    n_samples = 50
    with ThreadPoolExecutor(multiprocessing.cpu_count()) as pool:
        for _ in range(10):
            controls = np.zeros([n_samples, N_TIME, 20])

            t0 = perf_counter()
            parallel_rollout(pool, model, data, controls)
            dt = perf_counter() - t0

            print(f"| {n_samples} | {dt:.3f} |")


if __name__ == "__main__":
    main()
