#!/usr/bin/env python3
import mujoco
import numpy as np
from transformations import quaternion_from_euler

from arc_utilities import ros_init
from mjregrasping.goals import ObjectPointGoal
from mjregrasping.magnetic_fields import load_skeletons
from mjregrasping.run_evaluation import Runner
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.scenarios import conq_hose
from mjregrasping.settle import settle


class ConqHose(Runner):

    def __init__(self):
        super().__init__(conq_hose)

    def get_skeletons(self):
        return load_skeletons(conq_hose.skeletons_path)


@ros_init.with_ros("conq_hose")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    runner = ConqHose()
    runner.run([1])


if __name__ == "__main__":
    main()
