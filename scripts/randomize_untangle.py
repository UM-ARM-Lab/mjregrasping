#!/usr/bin/env python3
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import rerun as rr

from arc_utilities import ros_init
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.physics import Physics, get_q
from mjregrasping.trials import save_trial
from mjregrasping.rerun_visualizer import log_skeletons
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.scenarios import val_untangle, get_untangle_skeletons, setup_untangle
from mjregrasping.viz import make_viz, Viz


def randomize_rack(m: mujoco.MjModel, rng: np.random.RandomState):
    m.body("attach").pos[0] += rng.uniform(-0.05, 0.05)
    m.body("attach").pos[1] += rng.uniform(-0.05, 0.05)

    # shrink the computer rack in the X axis
    # h will be the half-size in X of the bottom/top geoms of the rack
    h = rng.uniform(0.2, 0.45)
    m.geom("rack1_bottom").size[1] = h
    m.geom("rack2_bottom").size[1] = h
    m.geom("rack2_top").size[1] = h + m.geom("rack1_post1").size[1]
    m.geom("rack1_post1").pos[1] = -h
    m.geom("rack1_post2").pos[1] = h
    m.geom("rack1_post3").pos[1] = -h
    m.geom("rack1_post4").pos[1] = h
    m.geom("rack2_post1").pos[1] = -h
    m.geom("rack2_post2").pos[1] = h
    m.geom("rack2_post3").pos[1] = -h
    m.geom("rack2_post4").pos[1] = h

    # compute the extents of the rack for placing the box and computer on it
    x0 = m.geom("rack1_bottom").pos[0] - m.geom("rack1_bottom").size[0] / 2
    x1 = m.geom("rack1_bottom").pos[0] + m.geom("rack1_bottom").size[0] / 2
    y0 = m.geom("rack1_bottom").pos[1] - m.geom("rack1_bottom").size[1] / 2
    y1 = m.geom("rack1_bottom").pos[1] + m.geom("rack1_bottom").size[1] / 2

    # place the box on the rack
    box1_x0 = x0 + m.geom("box1").size[0] / 2
    box1_x1 = x1 - m.geom("box1").size[0] / 2
    box1_y0 = y0 + m.geom("box1").size[1] / 2
    box1_y1 = y1 - m.geom("box1").size[1] / 2
    m.geom("box1").pos[0] = rng.uniform(box1_x0, box1_x1)
    m.geom("box1").pos[1] = rng.uniform(box1_y0, box1_y1)

    # place the computer on the rack
    c_x0 = x0 + m.geom("case").size[0] / 2
    c_x1 = x1 - m.geom("case").size[0] / 2
    c_y0 = y0 + m.geom("case").size[1] / 2
    c_y1 = y1 - m.geom("case").size[1] / 2
    m.body("computer").pos[0] = rng.uniform(c_x0, c_x1)
    m.body("computer").pos[1] = rng.uniform(c_y0, c_y1)


def randomize_qpos(phy: Physics, rng: np.random.RandomState, viz: Optional[Viz]):
    q = get_q(phy)
    q[0] += np.deg2rad(rng.uniform(-5, 5))
    q[1] += np.deg2rad(rng.uniform(-5, 5))
    pid_to_joint_config(phy, viz, q, sub_time_s=DEFAULT_SUB_TIME_S)


@ros_init.with_ros("randomize_untangle")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('randomize_untangle')
    rr.connect()

    scenario = val_untangle

    viz = make_viz(scenario)

    root = Path("trial_data") / scenario.name
    root.mkdir(exist_ok=True, parents=True)

    rng = np.random.RandomState(0)
    for i in range(10):
        m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))

        # Configure the model before we construct the data and physics object
        randomize_rack(m, rng)
        d = mujoco.MjData(m)
        objects = MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
        phy = Physics(m, d, objects)
        mujoco.mj_forward(phy.m, phy.d)
        viz.viz(phy, False)

        skeletons = get_untangle_skeletons(phy)
        log_skeletons(skeletons)

        setup_untangle(phy, viz)
        randomize_qpos(phy, rng, viz)

        save_trial(i, phy, scenario, None, skeletons)


if __name__ == "__main__":
    main()
