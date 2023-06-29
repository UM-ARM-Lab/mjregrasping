#!/usr/bin/env python3
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from time import perf_counter
import mujoco
import argparse

from mjregrasping.physics import Physics
from mjregrasping.rollout import parallel_rollout, DEFAULT_SUB_TIME_S

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xml_path", type=str)

    args = parser.parse_args()

    m = mujoco.MjModel.from_xml_path(args.xml_path)
    d = mujoco.MjData(m)
    phy = Physics(m, d)
    m.opt.timestep = 0.005

    n_samples = 50
    with ThreadPoolExecutor(multiprocessing.cpu_count() - 1) as pool:
        horizon = 10
        controls = np.zeros([n_samples, horizon, 20])
        sub_time_s = 0.02
        parallel_rollout(pool, phy, controls, sub_time_s)
        dts = []
        for _ in range(25):
            t0 = perf_counter()
            parallel_rollout(pool, phy, controls, sub_time_s)
            dt = perf_counter() - t0
            dts.append(dt)

            print(f"| {n_samples} | {dt:.3f} |")
        print(f"mean: {np.mean(dts):.3f} | std: {np.std(dts):.3f}")


if __name__ == "__main__":
    main()
