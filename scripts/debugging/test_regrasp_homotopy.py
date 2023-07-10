#!/usr/bin/env python3
import argparse
import time
from copy import copy
from pathlib import Path

import mujoco
import numpy as np
import pysdf_tools
import rerun as rr
from PIL import Image

from arc_utilities import ros_init
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.first_order_generator import FirstOrderGenerator
from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.goals import ObjectPointGoal
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.magnetic_fields import load_skeletons
from mjregrasping.mjsaver import load_data_and_eq
from mjregrasping.movie import MjRenderer
from mjregrasping.mujoco_objects import Objects
from mjregrasping.params import Params, hp
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rerun_visualizer import log_skeletons
from mjregrasping.rviz import MjRViz
from mjregrasping.scenarios import conq_hose
from mjregrasping.viz import Viz


@ros_init.with_ros("test_regrasp_homotopy")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rr.init('test_regrasp_homotopy')
    rr.connect()

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    scenario = conq_hose
    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))

    tfw = TF2Wrapper()
    mjviz = MjRViz(scenario.xml_path, tfw)
    p = Params()

    viz = Viz(rviz=mjviz, mjrr=MjReRun(scenario.xml_path), tfw=tfw, p=p)
    phy = Physics(m, mujoco.MjData(m),
                  objects=Objects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))
    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)

    goal = ObjectPointGoal(np.array([1.0, 0.0, 2.0]), 0.05, 1, viz)
    skeletons = load_skeletons(scenario.skeletons_path)
    sdf = pysdf_tools.SignedDistanceField.LoadFromFile(str(scenario.sdf_path))
    # viz.sdf(sdf, frame_id='world', idx=0)  # Very slow!

    # h = HomotopyGenerator(goal, skeletons, viz)
    h = FirstOrderGenerator(goal, sdf, viz)
    r = MjRenderer(m)

    states_dir = Path("states/conq_hose")
    for state_path in states_dir.glob("*.pkl"):
        d = load_data_and_eq(m, True, state_path)
        phy = Physics(m, d, objects=Objects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))
        viz.viz(phy)

        img = r.render(d)
        img_path = (state_path.parent / state_path.stem).with_suffix(".png")
        Image.fromarray(img).save(img_path)
        for _ in range(3):
            viz.viz(phy)
            log_skeletons(skeletons, color=(0, 255, 0, 255), timeless=True)
            time.sleep(0.01)

        from time import perf_counter
        t0 = perf_counter()
        result = h.generate(phy)
        print(f'H generate(): {perf_counter() - t0:.3f}s')

        if result is None:
            print("Homotopy not useful!")
            continue

        locs, subgoals = result

        _, _, xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy, locs)
        path1 = copy(get_rope_points(phy))
        for _ in range(3):
            for name, xpos_i, subgoal in zip(phy.o.rd.rope_grasp_eqs, xpos, subgoals):
                viz.sphere(f'{name}_xpos', xpos_i, radius=hp['grasp_goal_radius'], color=(0, 1, 0, 0.4),
                           frame_id='world', idx=0)
                viz.sphere(f'{name}_subgoal', subgoal, radius=goal.goal_radius, color=(0, 1, 0.5, 0.4),
                           frame_id='world', idx=0)
                path2 = np.stack([
                    path1[0],
                    subgoal,
                    path1[-1]
                ], axis=0)
                viz.lines(path2, 'first_order/path2', 0, 0.005, 'orange')
                time.sleep(0.001)


if __name__ == "__main__":
    main()
