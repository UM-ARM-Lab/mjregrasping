#!/usr/bin/env python3

import argparse

import numpy as np

from mjregrasping.initialize import initialize
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.rollout import control_step


def main():
    np.set_printoptions(precision=4, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("xml_path", type=str)

    args = parser.parse_args()

    model, data, mjviz, viz_pubs = initialize("test_move_to_joint_config", args.xml_path)

    # get the upper and lower actuator limits
    low = model.jnt_range[:, 0] * 0.2
    high = model.jnt_range[:, 1] * 0.2

    rng = np.random.RandomState(0)

    qvel_target = np.zeros(model.nu)
    for i in range(10):
        q_target = rng.uniform(low, high, size=model.nq)
        try:
            pid_to_joint_config(mjviz, model, data, q_target)
        except RuntimeError as e:
            if len(data.contact) > 0:
                # we hit the obstacle, continue to the next random configuration
                continue
            else:
                raise e


if __name__ == "__main__":
    main()
