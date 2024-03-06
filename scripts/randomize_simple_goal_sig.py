#!/usr/bin/env python3
from pathlib import Path
from typing import Optional

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
from mjregrasping.scenarios import threading_cable, get_threading_skeletons, dz, dy, simple_goal_sig
from mjregrasping.settle import settle
from mjregrasping.trials import save_trial
from mjregrasping.viz import make_viz, Viz


def randomize_loop_positions(original_path: Path, rng: np.random.RandomState):
    mxml = MujocoXmlExpander(original_path)

    dy = 0.15
    loop1 = mxml.get_e("body", "loop1")
    loop1_pos = mxml.get_vec(loop1, 'pos')
    loop1_pos[1] += rng.uniform(-dy, dy)
    mxml.set_vec(loop1, loop1_pos, 'pos')

    tmp_path = mxml.save_tmp()
    m = mujoco.MjModel.from_xml_path(str(tmp_path))
    return m


@ros_init.with_ros("randomize_simple_goal_sig")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('randomize_simple_goal_sig')
    rr.connect()

    scenario = simple_goal_sig

    viz = make_viz(scenario)

    root = Path("trial_data") / scenario.name
    root.mkdir(exist_ok=True, parents=True)

    rng = np.random.RandomState(0)
    for i in range(0, 5):
        # Configure the model before we construct the data and physics object
        m = randomize_loop_positions(scenario.xml_path, rng)

        d = mujoco.MjData(m)
        objects = MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
        phy = Physics(m, d, objects)

        mujoco.mj_forward(m, d)

        rope_xyz_q_indices = phy.o.rope.qpos_indices[:3]
        rope_quat_q_indices = phy.o.rope.qpos_indices[3:7]
        phy.d.qpos[rope_xyz_q_indices] = phy.d.geom("attach1").xpos
        phy.d.qpos[rope_quat_q_indices] = quaternion_from_euler(0, 0, 3.4)

        mujoco.mj_forward(m, d)
        viz.viz(phy)

        activate_grasp(phy, 'attach1', 0.0)
        settle(phy, DEFAULT_SUB_TIME_S, viz, is_planning=False)

        q = np.array([
            0, -0.8,  # torso
            0, 0.0, 0.0, 0.0, 0, 0, 0,  # left arm
            0,  # left gripper
            0, 0.0, 0, 0.0, 0, -0.0, -1.0,  # right arm
            0,  # right gripper
        ])
        pid_to_joint_config(phy, viz, q, sub_time_s=DEFAULT_SUB_TIME_S)
        q[0] = rng.uniform(-0.4, 0.0)
        q[1] = rng.uniform(-0.4, -0.1)
        pid_to_joint_config(phy, viz, q, sub_time_s=DEFAULT_SUB_TIME_S)

        if viz:
            viz.viz(phy)

        skeletons = {
            "loop1": np.array([
                d.geom("loop1_front").xpos - dz(m.geom("loop1_front").size[2]),
                d.geom("loop1_front").xpos + dz(m.geom("loop1_front").size[2]),
                d.geom("loop1_back").xpos + dz(m.geom("loop1_back").size[2]),
                d.geom("loop1_back").xpos - dz(m.geom("loop1_back").size[2]),
                d.geom("loop1_front").xpos - dz(m.geom("loop1_front").size[2]),
            ])
        }

        save_trial(i, phy, scenario, None, skeletons)
        print(f"Saved trial {i}")


if __name__ == "__main__":
    main()
