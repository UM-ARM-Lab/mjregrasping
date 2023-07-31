#!/usr/bin/env python3
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
import pysdf_tools
import rerun as rr
from PIL import Image

from arc_utilities import ros_init
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.homotopy_utils import load_skeletons
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner, params_to_locs_and_subgoals
from mjregrasping.mjsaver import load_data_and_eq
from mjregrasping.movie import MjRenderer
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.params import Params
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rerun_visualizer import log_skeletons
from mjregrasping.rviz import MjRViz
from mjregrasping.scenarios import val_untangle, make_untangle_goal
from mjregrasping.sdf_collision_checker import SDFCollisionChecker
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
                  objects=MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))
    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)

    skeletons = load_skeletons(scenario.skeletons_path)
    sdf = pysdf_tools.SignedDistanceField.LoadFromFile(str(scenario.sdf_path))
    cc = SDFCollisionChecker(sdf)

    goal = make_untangle_goal(viz)
    r = MjRenderer(m)

    states_dir = Path("states/untangle")
    rr.set_time_sequence('homotopy', 0)
    states_paths = [
        Path("states/Untangle/1690492695.pkl"),
        # Path('states/untangle/debugging3.pkl'),
        # Path('states/untangle/debugging2.pkl'),
        # Path('states/untangle/debugging.pkl'),
        # Path('states/untangle/1688067719.pkl'),
        # Path('states/untangle/1688067882.pkl'),
        # Path('states/untangle/1688071991.pkl'),
        # Path('states/untangle/1688072614.pkl'),
        # Path('states/untangle/1688067686.pkl'),
    ]
    for state_path in states_paths:
        for seed in range(1, 3):
            h = HomotopyRegraspPlanner(goal, skeletons, cc, seed=seed)
            d = load_data_and_eq(m, state_path, True)
            phy = Physics(m, d, objects=MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))

            viz.viz(phy)
            rr.log_cleared("planned", recursive=True)
            log_skeletons(skeletons, color=(0, 255, 0, 255), timeless=True)

            img = r.render(d)
            img_path = (state_path.parent / state_path.stem).with_suffix(".png")
            Image.fromarray(img).save(img_path)

            t0 = perf_counter()
            params, strategy = h.generate_params(phy, viz=viz, viz_ik=True)
            t1 = perf_counter()

            locs, subgoals, _ = params_to_locs_and_subgoals(phy, strategy, params)
            _, _, xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy, locs)

            tool_paths = np.concatenate((xpos[:, None], subgoals), axis=1)
            for tool_name, path in zip(phy.o.rd.tool_sites, tool_paths):
                viz.lines(path, f'homotopy/{tool_name}_path_sln', idx=0, scale=0.02, color=[0, 0, 1, 0.5])
            cost = h.cost(strategy, phy, viz, **params, viz_ik=True, viz_loops=True)
            print(f'H generate(): {t1 - t0:.3f}s {cost=:} {locs=:}')
            print()


if __name__ == "__main__":
    main()
