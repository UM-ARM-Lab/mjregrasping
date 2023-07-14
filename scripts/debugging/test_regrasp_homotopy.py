#!/usr/bin/env python3
from pathlib import Path

import mujoco
import numpy as np
import pysdf_tools
import rerun as rr
from PIL import Image

from arc_utilities import ros_init
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.homotopy_regrasp_generator import HomotopyGenerator
from mjregrasping.magnetic_fields import load_skeletons
from mjregrasping.mjsaver import load_data_and_eq
from mjregrasping.movie import MjRenderer
from mjregrasping.mujoco_objects import Objects
from mjregrasping.params import Params
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rerun_visualizer import log_skeletons
from mjregrasping.rviz import MjRViz
from mjregrasping.scenarios import val_untangle, make_untangle_goal
from mjregrasping.viz import Viz


@ros_init.with_ros("test_regrasp_homotopy")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rr.init('test_regrasp_homotopy')
    rr.connect()

    scenario = val_untangle
    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))

    tfw = TF2Wrapper()
    mjviz = MjRViz(scenario.xml_path, tfw)
    p = Params()

    viz = Viz(rviz=mjviz, mjrr=MjReRun(scenario.xml_path), tfw=tfw, p=p)
    phy = Physics(m, mujoco.MjData(m),
                  objects=Objects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))
    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)

    skeletons = load_skeletons(scenario.skeletons_path)
    sdf = pysdf_tools.SignedDistanceField.LoadFromFile(str(scenario.sdf_path))
    viz.sdf(sdf, frame_id='world', idx=0)

    goal = make_untangle_goal(viz)
    h = HomotopyGenerator(goal, skeletons, sdf, seed=4)
    r = MjRenderer(m)

    states_dir = Path("states/untangle")
    rr.set_time_sequence('homotopy', 0)
    for state_path in sorted(states_dir.glob("*.pkl")):
        # state_path = states_dir / '1688067882.pkl'
        d = load_data_and_eq(m, True, state_path)
        phy = Physics(m, d, objects=Objects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))

        viz.viz(phy)
        log_skeletons(skeletons, color=(0, 255, 0, 255), timeless=True)

        img = r.render(d)
        img_path = (state_path.parent / state_path.stem).with_suffix(".png")
        Image.fromarray(img).save(img_path)

        from time import perf_counter
        t0 = perf_counter()
        result = h.generate(phy, viz=viz)
        locs, subgoals = result
        t1 = perf_counter()

        _, _, xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy, locs)
        tool_paths = np.concatenate((xpos[:, None], subgoals), axis=1)
        for tool_name, path in zip(phy.o.rd.tool_sites, tool_paths):
            viz.lines(path, f'homotopy/{tool_name}_path', idx=0, scale=0.02, color=[0, 0, 1, 0.5])
        print(f'H generate(): {t1 - t0:.3f}s {locs=:}')
        print()


if __name__ == "__main__":
    main()
