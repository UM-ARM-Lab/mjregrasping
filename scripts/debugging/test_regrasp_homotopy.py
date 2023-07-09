#!/usr/bin/env python3
import argparse
import pickle
import time
from pathlib import Path

import mujoco
import numpy as np
import rerun as rr
from PIL import Image

from arc_utilities import ros_init
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.goals import ObjectPointGoal
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.homotopy_regrasp_generator import HomotopyGenerator, HomotopyType
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
from sdf_tools.utils_3d import compute_sdf


@ros_init.with_ros("test_regrasp_homotopy")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rr.init('test_regrasp_homotopy')
    rr.connect()

    parser = argparse.ArgumentParser()
    parser.add_argument("homotopy_type", type=str, choices=['TRUE', 'FIRST_ORDER'])

    args = parser.parse_args()
    homotopy_type = HomotopyType[args.homotopy_type]

    scenario = conq_hose
    m = mujoco.MjModel.from_xml_path(scenario.xml_path)

    tfw = TF2Wrapper()
    mjviz = MjRViz(scenario.xml_path, tfw)
    p = Params()

    viz = Viz(rviz=mjviz, mjrr=MjReRun(scenario.xml_path), tfw=tfw, p=p)

    goal = ObjectPointGoal(np.array([1.0, 0.0, 2.0]), 0.05, 1, viz)
    skeletons = load_skeletons(scenario.skeletons_path)
    sdf = load_sdf(scenario.vg_path)
    h = HomotopyGenerator(goal, sdf, skeletons, viz)
    r = MjRenderer(m)

    states_dir = Path("states/conq_hose")
    for state_path in states_dir.glob("*.pkl"):
        d = load_data_and_eq(m, True, state_path)
        phy = Physics(m, d, objects=Objects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))

        img = r.render(d)
        img_path = (state_path.parent / state_path.stem).with_suffix(".png")
        Image.fromarray(img).save(img_path)
        for _ in range(3):
            viz.viz(phy)
            log_skeletons(skeletons, color=(0, 255, 0, 255), timeless=True)
            time.sleep(0.01)

        from time import perf_counter
        t0 = perf_counter()
        locs = h.generate(phy, homotopy_type)
        print(f'dt: {perf_counter() - t0:.4f}')
        if locs is None:
            print("Homotopy not useful!")
        _, _, xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy, locs)
        for _ in range(3):
            for name, xpos_i in zip(phy.o.rd.rope_grasp_eqs, xpos):
                viz.sphere(f'{name}_xpos', xpos_i, radius=hp['grasp_goal_radius'], color=(0, 1, 0, 0.4),
                           frame_id='world', idx=0)
                time.sleep(0.001)


def load_sdf(vg_path: Path):
    with vg_path.open('rb') as f:
        vg = pickle.load(f)
    sdf = compute_sdf(vg.vg, vg.res, vg.origin_point)
    return sdf


if __name__ == "__main__":
    main()
