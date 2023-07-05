#!/usr/bin/env python3
import time
from pathlib import Path

import mujoco
import numpy as np
import rerun as rr

from arc_utilities import ros_init
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.goals import ObjectPointGoal
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.mjsaver import load_data_and_eq
from mjregrasping.movie import MjRenderer
from mjregrasping.params import Params
from mjregrasping.physics import Physics
from mjregrasping.slack_regrasp_generator import SlackGenerator
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rviz import MjRViz
from mjregrasping.viz import Viz


@ros_init.with_ros("test_regrasp_slack")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rr.init('test_regrasp_slack')
    rr.connect()

    xml_path = "models/pull_scene.xml"
    m = mujoco.MjModel.from_xml_path(xml_path)

    tfw = TF2Wrapper()
    mjviz = MjRViz(xml_path, tfw)
    p = Params()

    viz = Viz(rviz=mjviz, mjrr=MjReRun(xml_path), tfw=tfw, p=p)

    r = MjRenderer(m)

    states_dir = Path("states/pull")
    for state_path in states_dir.glob("*.pkl"):
        d = load_data_and_eq(m, True, state_path)
        phy = Physics(m, d, "floor")

        for _ in range(3):
            viz.viz(phy)

        for goal_loc in [0, 1]:
            goal = ObjectPointGoal(np.array([1.0, 0.0, 2.0]), 0.05, goal_loc, viz)
            h = SlackGenerator(goal, viz)
            from time import perf_counter
            t0 = perf_counter()
            locs = h.generate(phy)
            print(f'dt: {perf_counter() - t0:.4f}')
            print(locs)
            if locs is None:
                print("Homotopy not useful!")
            _, _, xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy, locs)
            for _ in range(3):
                viz.sphere('left_slack_xpos', xpos[0] + 1e-3, radius=0.02, color=(0, 1., 0, 0.5), frame_id='world', idx=0)
                viz.sphere('right_slack_xpos', xpos[1], radius=0.02, color=(0, 0., 1., 0.5), frame_id='world', idx=0)
                time.sleep(0.01)


if __name__ == "__main__":
    main()
