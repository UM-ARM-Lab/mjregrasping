#!/usr/bin/env python3
from pathlib import Path

import mujoco
import numpy as np
import rerun as rr
from transformations import quaternion_from_euler

from arc_utilities import ros_init
from mjregrasping.grasping import activate_grasp
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.physics import Physics
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.rviz import MujocoXmlExpander
from mjregrasping.scenarios import threading_cable, get_threading_skeletons
from mjregrasping.settle import settle
from mjregrasping.trials import save_trial
from mjregrasping.viz import make_viz, Viz


def randomize_loop_positions(original_path: Path, rng: np.random.RandomState):
    mxml = MujocoXmlExpander(original_path)

    dy = 0.06
    dz = 0.1

    loop1 = mxml.get_e("body", "loop1")
    loop1_pos = mxml.get_vec(loop1, 'pos')
    loop1_pos[1] += rng.uniform(-dy, dy)
    loop1_pos[2] += rng.uniform(-dz, dz)
    mxml.set_vec(loop1, loop1_pos, 'pos')

    loop2 = mxml.get_e("body", "loop2")
    loop2_pos = mxml.get_vec(loop2, 'pos')
    loop2_pos[1] += rng.uniform(-dy, dy)
    loop2_pos[2] += rng.uniform(-dz, dz)
    mxml.set_vec(loop2, loop2_pos, 'pos')

    loop3 = mxml.get_e("body", "loop3")
    loop3_pos = mxml.get_vec(loop3, 'pos')
    loop3_pos[1] += rng.uniform(-dy, dy)
    loop3_pos[2] += rng.uniform(-dz, dz)
    mxml.set_vec(loop3, loop3_pos, 'pos')

    tmp_path = mxml.save_tmp()
    m = mujoco.MjModel.from_xml_path(str(tmp_path))
    return m



@ros_init.with_ros("randomize_threading")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('randomize_threading')
    rr.connect()

    scenario = threading_cable

    viz = make_viz(scenario)

    root = Path("trial_data") / scenario.name
    root.mkdir(exist_ok=True, parents=True)

    rng = np.random.RandomState(0)
    for i in range(3, 25):
        # Configure the model before we construct the data and physics object
        m = randomize_loop_positions(scenario.xml_path, rng)

        d = mujoco.MjData(m)
        objects = MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
        phy = Physics(m, d, objects)

        rope_xyz_q_indices = phy.o.rope.qpos_indices[:3]
        rope_quat_q_indices = phy.o.rope.qpos_indices[3:7]
        phy.d.qpos[rope_xyz_q_indices] = np.array([1.5, 0.6, 0.0])
        phy.d.qpos[rope_quat_q_indices] = quaternion_from_euler(0, 0.2, np.pi)

        activate_grasp(phy, 'attach1', 0.0)

        q = np.array([
            0.4, 0.8,  # torso
            0.0, 0.0, 0.0, 0.0, 0, 0, 0,  # left arm
            0,  # left gripper
            0.0, 0.0, 0, 0.0, 0, -0.0, -0.5,  # right arm
            0.4,  # right gripper
        ])
        pid_to_joint_config(phy, viz, q, sub_time_s=DEFAULT_SUB_TIME_S)
        activate_grasp(phy, 'right', 0.93)
        settle(phy, DEFAULT_SUB_TIME_S, viz, is_planning=False)
        q = np.array([
            rng.uniform(-0.7, 0), rng.uniform(-0.2, 0.2),  # torso
            -0.2, 0.0, 0.0, 0.0, 0, 0, 0,  # left arm
            0,  # left gripper
            -0.1, 0.0, 0, 0.0, 0, -0.0, -1.0,  # right arm
            0.06,  # right gripper
        ])
        pid_to_joint_config(phy, viz, q, sub_time_s=DEFAULT_SUB_TIME_S)

        loc = rng.uniform(0.93, 0.96)
        activate_grasp(phy, 'right', loc)
        settle(phy, DEFAULT_SUB_TIME_S, viz, is_planning=False)

        if viz:
            viz.viz(phy)

        skeletons = get_threading_skeletons(phy)
        viz.skeletons(skeletons)

        save_trial(i, phy, scenario, None, skeletons)
        print(f"Saved trial {i}")


if __name__ == "__main__":
    main()
