#!/usr/bin/env python3
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import networkx as nx
import numpy as np
import rerun as rr
from PIL import Image

from arc_utilities import ros_init
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.magnetic_fields import load_skeletons
from mjregrasping.mjsaver import load_data_and_eq
from mjregrasping.movie import MjRenderer
from mjregrasping.mujoco_objects import Objects
from mjregrasping.params import Params
from mjregrasping.path_comparer import TrueHomotopyComparer, NO_HOMOTOPY
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rerun_visualizer import log_skeletons
from mjregrasping.rviz import MjRViz
from mjregrasping.scenarios import cable_harness
from mjregrasping.viz import Viz


@ros_init.with_ros("make_homotopy_results")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rr.init('make_homotopy_results')
    rr.connect()

    # scenario = val_untangle
    scenario = cable_harness
    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))

    tfw = TF2Wrapper()
    mjviz = MjRViz(scenario.xml_path, tfw)
    p = Params()

    viz = Viz(rviz=mjviz, mjrr=MjReRun(scenario.xml_path), tfw=tfw, p=p)
    objects = Objects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
    phy = Physics(m, mujoco.MjData(m), objects)
    mujoco.mj_forward(phy.m, phy.d)

    r = MjRenderer(m)

    skeletons = load_skeletons(scenario.skeletons_path)
    log_skeletons(skeletons, color=(0, 255, 0, 255), timeless=True, stroke_width=0.02)

    comparer = TrueHomotopyComparer(skeletons)

    states_dir = Path(f"states/{scenario.name}")
    states_paths = list(states_dir.glob("*.pkl"))
    n_states = len(states_paths)
    # Divide n_states into rows and cols in way that looks nice
    nrows = int(np.ceil(np.sqrt(n_states)))
    ncols = int(np.ceil(n_states / nrows))

    results = []
    for i, state_path in enumerate(states_paths):
        d = load_data_and_eq(m, True, state_path)
        phy = Physics(m, d, objects=Objects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))

        img = np.copy(r.render(d))

        # Visualize the robot in rerun
        viz.rviz.viz(phy, is_planning=False)
        viz.mjrr.viz(phy, is_planning=False, detailed=True)

        initial_rope_points = copy(get_rope_points(phy))
        graph = comparer.create_graph_nodes(phy)
        arm_points = comparer.get_arm_points(phy)
        comparer.add_edges(graph, initial_rope_points, arm_points)
        h = comparer.get_signature(phy, initial_rope_points, log_loops=True)

        # TODO: add the nx graphs, and a easier to interpret image of the loops & skeletons
        nx_fig, nx_ax = plt.subplots()
        loc_labels = {k: f'{k}\nloc={v:.1f}' for k, v in nx.get_node_attributes(graph, 'loc').items()}
        nx.draw(graph, labels=loc_labels, node_size=5000, ax=nx_ax)
        nx_fig.show()
        nx_fig.savefig(states_dir / f"{state_path.stem}_nx.pdf", dpi=300, format="pdf")
        nx_fig.savefig(states_dir / f"{state_path.stem}_nx.png", dpi=300)

        img_path = states_dir / f"{state_path.stem}_mj.png"
        Image.fromarray(img).save(img_path)

        results.append({
            'h': h,
            'mj_img': img,
            'nx_img': nx_fig,
        })

    plt.style.use('paper')
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 10, nrows * 6))
    results = sorted(results, key=lambda r: r['h'])
    for i, result in enumerate(results):
        h = result['h']
        if h == NO_HOMOTOPY:
            h = '∅'
        else:
            h = sorted(h)
        mj_img = result['mj_img']

        row = i // ncols
        col = i % ncols
        ax = axes[row, col]

        ax.imshow(mj_img)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f"$\mathcal{{H}}=${h}")

    # Remove any extra unused axes
    for i in range(n_states, nrows * ncols):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        ax.axis('off')

    plt.savefig("results/homotopy_results.pdf", dpi=300, format="pdf")
    fig.show()


if __name__ == "__main__":
    main()
