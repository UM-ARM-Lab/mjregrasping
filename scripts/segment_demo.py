import argparse
import pickle
from pathlib import Path

import numpy as np
import rerun as rr

import rospy
from mjregrasping.scenarios import cable_harness
from mjregrasping.segment_demo import load_demo, viz_subgoals_by_h, viz_subgoals_by_locs, save_cached_demo
from mjregrasping.viz import make_viz


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rospy.init_node("segment_demo")
    parser = argparse.ArgumentParser()
    parser.add_argument("demo", type=Path)
    args = parser.parse_args()

    assert args.demo.is_dir()

    scenario = cable_harness

    rr.init("viewer")
    rr.connect()

    viz = make_viz(scenario)

    hs, locs_seq, phys, paths = load_demo(args.demo, scenario)
    save_cached_demo(args.demo, hs, locs_seq, paths, phys)

    viz_subgoals_by_h(phys, hs, paths, viz)
    # segment_by_locs(phys, locs_seq, viz)


if __name__ == '__main__':
    main()
