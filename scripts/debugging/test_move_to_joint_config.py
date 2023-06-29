#!/usr/bin/env python3

import argparse

import numpy as np

from mjregrasping.initialize import initialize
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.rollout import DEFAULT_SUB_TIME_S


def main():
    np.set_printoptions(precision=4, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("xml_path", type=str)

    args = parser.parse_args()

    m, d, viz = initialize("test_move_to_joint_config", args.xml_path)

    # get the upper and lower actuator limits
    low = m.jnt_range[:, 0] * 0.2
    high = m.jnt_range[:, 1] * 0.2

    rng = np.random.RandomState(0)

    for i in range(10):
        q_target = rng.uniform(low, high, size=m.nq)
        try:
            pid_to_joint_config(viz, m, d, q_target, sub_time_s=DEFAULT_SUB_TIME_S)
        except RuntimeError as e:
            if len(d.contact) > 0:
                # we hit the obstacle, continue to the next random configuration
                continue
            else:
                raise e


if __name__ == "__main__":
    main()
