#!/usr/bin/env python3
import time
from pathlib import Path

import mujoco
import numpy as np
import rerun as rr
from PIL import Image

import rospy
from arc_utilities import ros_init
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.goals import ObjectPointGoal
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.magnetic_fields import load_skeletons
from mjregrasping.mjsaver import load_data_and_eq
from mjregrasping.movie import MjRenderer
from mjregrasping.params import Params
from mjregrasping.physics import Physics
from mjregrasping.homotopy_regrasp_generator import HomotopyGenerator
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rerun_visualizer import log_skeletons
from mjregrasping.rviz import MjRViz
from mjregrasping.viz import Viz


@ros_init.with_ros("test_regrasp_homotopy")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rr.init('test_regrasp_homotopy')
    rr.connect()

    xml_path = "models/untangle_scene.xml"
    m = mujoco.MjModel.from_xml_path(xml_path)

    tfw = TF2Wrapper()
    mjviz = MjRViz(xml_path, tfw)
    p = Params()

    viz = Viz(rviz=mjviz, mjrr=MjReRun(xml_path), tfw=tfw, p=p)

    goal = ObjectPointGoal(np.array([1.0, 0.0, 2.0]), 0.05, 1, viz)
    skeletons = load_skeletons("models/computer_rack_skeleton.hjson")
    h = HomotopyGenerator(goal, skeletons, viz)
    r = MjRenderer(m)

    states_dir = Path("states/untangle")
    for state_path in states_dir.glob("*.pkl"):
        # state_path = states_dir / "1688067719.pkl"
        d = load_data_and_eq(m, True, state_path)
        phy = Physics(m, d, "computer_rack")

        img = r.render(d)
        img_path = (state_path.parent / state_path.stem).with_suffix(".png")
        Image.fromarray(img).save(img_path)
        for _ in range(3):
            viz.viz(phy)
            log_skeletons(skeletons, color=(0, 255, 0, 255), timeless=True)
            time.sleep(0.01)

        from time import perf_counter
        t0 = perf_counter()
        locs = h.generate(phy)
        print(f'dt: {perf_counter() - t0:.4f}')
        if locs is None:
            print("Homotopy not useful!")
        _, _, xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy, locs)
        for _ in range(3):
            viz.sphere('left_homotopy_xpos', xpos[0], radius=0.02, color='g', frame_id='world', idx=0)
            viz.sphere('right_homotopy_xpos', xpos[1], radius=0.02, color='b', frame_id='world', idx=0)
            time.sleep(0.01)

        # paths = h.find_rope_robot_paths(phy)
        # if paths is None:
        #     print("Homotopy not useful!")
        # else:
        #     for _ in range(3):
        #         for i, path in enumerate(paths):
        #             viz.lines(ns='path', positions=path, idx=0, scale=0.01, color=cm.jet(i / len(paths)))
        #             time.sleep(0.01)
        # pass


if __name__ == "__main__":
    main()
