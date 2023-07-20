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
from mjregrasping.homotopy_checker import add_edges, create_graph_nodes, get_arm_points, get_full_h_signature, \
    compare_to_goal, compare_h_signature_to_goal
from mjregrasping.homotopy_utils import load_skeletons, NO_HOMOTOPY
from mjregrasping.mjsaver import load_data_and_eq
from mjregrasping.movie import MjRenderer
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.params import Params
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
    objects = MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
    phy = Physics(m, mujoco.MjData(m), objects)
    mujoco.mj_forward(phy.m, phy.d)

    r = MjRenderer(m)

    skeletons = load_skeletons(scenario.skeletons_path)

    states_dir = Path(f"states/{scenario.name}")
    states_paths = list(states_dir.glob("*.pkl"))
    n_states = len(states_paths)
    # Divide n_states into rows and cols in way that looks nice
    nrows = int(np.ceil(np.sqrt(n_states)))
    ncols = int(np.ceil(n_states / nrows))

    # To demonstrate how homotopy can be used to compare a state to the goal state,
    # we load the goal state and compare it each of the other states.
    # The goal comparison is somewhat distinct from the comparison we do for grasping.
    goal_path = states_dir / "goal_state.pkl"
    d = load_data_and_eq(m, True, goal_path)
    phy = Physics(m, d, objects=MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))
    goal_rope_points = copy(get_rope_points(phy))

    results = []
    for i, state_path in enumerate(states_paths):
        d = load_data_and_eq(m, True, state_path)
        phy = Physics(m, d, objects=MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))

        img = np.copy(r.render(d))

        # Visualize the robot in rerun
        viz.rviz.viz(phy, is_planning=False)
        viz.mjrr.viz(phy, is_planning=False, detailed=True)
        log_skeletons(skeletons, color=(0, 255, 0, 255), stroke_width=0.02)

        rope_points = copy(get_rope_points(phy))
        graph = create_graph_nodes(phy)
        arm_points = get_arm_points(phy)
        add_edges(graph, rope_points, arm_points)
        h, loops = get_full_h_signature(skeletons, graph, rope_points, arm_points)

        goal_h_same = compare_h_signature_to_goal(skeletons, rope_points, goal_rope_points)
        goal_tol = 0.05  # distance between start/end pairs in meters
        start_same = np.linalg.norm(rope_points[0] - goal_rope_points[0]) < goal_tol
        end_same = np.linalg.norm(rope_points[-1] - goal_rope_points[-1]) < goal_tol
        goal_reached = goal_h_same and start_same and end_same

        rr.log_cleared(f'loops', recursive=True)
        for i, l in enumerate(loops):
            rr.log_line_strip(f'loops/{i}', l, stroke_width=0.02)

        # TODO: add the nx graphs, and a easier to interpret image of the loops & skeletons
        nx_fig, nx_ax = plt.subplots()
        loc_labels = {k: f'{k}\nloc={v:.1f}' for k, v in nx.get_node_attributes(graph, 'loc').items()}
        nx.draw(graph, labels=loc_labels, node_size=5000, ax=nx_ax, margins=(0.075, 0.075))
        nx_fig.savefig(states_dir / f"{state_path.stem}_nx.pdf", dpi=300, format="pdf")
        nx_img_path = states_dir / f"{state_path.stem}_nx.png"
        nx_fig.tight_layout(pad=0)
        nx_fig.savefig(nx_img_path, dpi=300)

        img_path = states_dir / f"{state_path.stem}_mj.png"
        Image.fromarray(img).save(img_path)

        results.append({
            'h':              h,
            'loops':          loops,
            'mj_img':         img,
            'nx_img_path':    nx_img_path,
            'start_end_same': start_same and end_same,
            'goal_h_same':    goal_h_same,
            'goal_reached':   goal_reached,
        })

    plt.style.use('paper')
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 10, nrows * 6))
    results = sorted(results, key=lambda r: len(str(r['h'])))
    for i, result in enumerate(results):
        h = result['h']
        if h == NO_HOMOTOPY:
            h = '∅'
        else:
            h = sorted(h)
        mj_img = result['mj_img']
        nx_img = Image.open(result['nx_img_path'])
        goal_h_same = result['goal_h_same']
        start_end_same = result['start_end_same']

        row = i // ncols
        col = i % ncols
        ax = axes[row, col]

        ax.imshow(mj_img)

        sub_ax_size = 0.35
        sub_ax = ax.inset_axes([1 - sub_ax_size, 1 - sub_ax_size, sub_ax_size, sub_ax_size])
        sub_ax.imshow(nx_img)
        sub_ax.axis("off")

        ax.axis("off")
        start_end_indicator = "✓" if start_end_same else "✗"
        goal_h_indicator = "✓" if goal_h_same else "✗"
        title_lines = [
            f"$\mathcal{{H}}=${h}",
            f"goal start/end reached: {start_end_indicator}",
            f"goal h same: {goal_h_indicator}"
        ]
        ax.set_title('\n'.join(title_lines))

    # turn off all the (extra) axes
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j].axis("off")

    plt.savefig("results/homotopy_results.pdf", dpi=300, format="pdf")
    fig.show()


if __name__ == "__main__":
    main()
