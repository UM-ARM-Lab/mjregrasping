import argparse
from pathlib import Path

import mujoco.viewer
import numpy as np
import rerun as rr

import rospy
from mjregrasping.grasping import get_grasp_locs
from mjregrasping.homotopy_checker import get_full_h_signature_from_phy
from mjregrasping.homotopy_utils import load_skeletons
from mjregrasping.mjsaver import load_data_and_eq
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.physics import Physics
from mjregrasping.scenarios import cable_harness
from mjregrasping.viz import make_viz


def print_hs(hs):
    for i, h in enumerate(hs):
        print(f'{i}: {h}')


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

    skeletons = load_skeletons(scenario.skeletons_path)

    paths = sorted(list(args.demo.glob("*.pkl")))
    print(f'Found {len(paths)} states in the demonstration.')
    hs = []
    locs_seq = []
    phys = []
    for path in paths:
        m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
        d = load_data_and_eq(m, path, True)
        phy = Physics(m, d, objects=MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))
        h, _ = get_full_h_signature_from_phy(skeletons, phy,
                                             collapse_empty_gripper_cycles=False,
                                             gripper_ids_in_h_signature=True)
        locs = get_grasp_locs(phy)
        locs_seq.append(locs)
        hs.append(h)
        phys.append(phy.copy_all())

    segment_by_h(phys, hs, viz)
    segment_by_locs(phys, locs_seq, viz)


def segment_by_locs(phys, locs_seq, viz):
    last_subgoal_locs = None
    subgoal_phys = []
    rr.set_time_sequence('subgoal_by_locs', 0)
    for locs, phy in zip(locs_seq, phys):
        if np.any(locs != last_subgoal_locs) or last_subgoal_locs is None:
            last_subgoal_locs = locs
            subgoal_phys.append(phy)
            rr.set_time_sequence('subgoal_by_locs', len(subgoal_phys))
            viz.viz(phy, is_planning=False)
            rr.log_text_entry('locs', f'{locs}')
    # also show final state
    rr.set_time_sequence('subgoal_by_locs', len(subgoal_phys) + 1)
    viz.viz(phy, is_planning=False)


def segment_by_h(phys, hs, viz):
    print_hs(hs)
    last_subgoal_h = None
    subgoal_phys = []
    rr.set_time_sequence('subgoal_by_h', 0)
    for h, phy in zip(hs, phys):
        if h != last_subgoal_h or last_subgoal_h is None:
            last_subgoal_h = h
            subgoal_phys.append(phy)
            rr.set_time_sequence('subgoal_by_h', len(subgoal_phys))
            viz.viz(phy, is_planning=False, detailed=True)
            rr.log_text_entry('h', f'{h}')
    # also show final state
    rr.set_time_sequence('subgoal_by_h', len(subgoal_phys) + 1)
    viz.viz(phy, is_planning=False, detailed=True)


if __name__ == '__main__':
    main()
