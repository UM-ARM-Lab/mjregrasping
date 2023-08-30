import argparse
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
import rerun as rr

import rospy
from mjregrasping.mjsaver import load_data_and_eq
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.physics import Physics
from mjregrasping.scenarios import threading
from mjregrasping.viz import make_viz


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rospy.init_node("viz_demo")
    parser = argparse.ArgumentParser()
    parser.add_argument("demo", type=Path)
    args = parser.parse_args()

    assert args.demo.is_dir()

    scenario = threading

    rr.init("viewer")
    rr.connect()

    viz = make_viz(scenario)

    paths = sorted(list(args.demo.glob("*.pkl")))
    print(f'Found {len(paths)} states in the demonstration.')
    for path in paths:
        m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
        d = load_data_and_eq(m, path, True)
        rr.log_text_entry("path", str(path))
        phy = Physics(m, d, objects=MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))
        viz.viz(phy, is_planning=False, detailed=True)


if __name__ == '__main__':
    main()
