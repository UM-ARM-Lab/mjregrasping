#!/usr/bin/env python3
import pickle
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import rerun as rr

from arc_utilities import ros_init
from mjregrasping.mjcf_scene_to_sdf import get_sdf
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.physics import Physics, get_q
from mjregrasping.trials import save_trial
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.scenarios import cable_harness, setup_cable_harness, get_cable_harness_skeletons
from mjregrasping.viz import make_viz, Viz


def randomize_loop_positions(m: mujoco.MjModel, rng: np.random.RandomState):
    m.body("loop1").pos[1] += rng.uniform(-0.1, 0.1)
    m.body("loop1").pos[2] += rng.uniform(-0.1, 0.1)
    m.body("loop2").pos[1] += rng.uniform(-0.1, 0.1)
    m.body("loop2").pos[2] += rng.uniform(-0.1, 0.1)
    m.body("loop3").pos[1] += rng.uniform(-0.1, 0.1)
    m.body("loop3").pos[2] += rng.uniform(-0.1, 0.1)


def randomize_qpos(phy: Physics, rng: np.random.RandomState, viz: Optional[Viz]):
    q = get_q(phy)
    q[0] += np.deg2rad(rng.uniform(-5, 5))
    q[1] += np.deg2rad(rng.uniform(-5, 5))
    pid_to_joint_config(phy, viz, q, sub_time_s=DEFAULT_SUB_TIME_S)


@ros_init.with_ros("randomize_cable_harness")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('randomize_cable_harness')
    rr.connect()

    scenario = cable_harness

    viz = make_viz(scenario)

    root = Path("trial_data") / scenario.name
    root.mkdir(exist_ok=True, parents=True)

    rng = np.random.RandomState(0)
    for i in range(10):
        m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))

        # Configure the model before we construct the data and physics object
        randomize_loop_positions(m, rng)

        d = mujoco.MjData(m)
        objects = MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
        phy = Physics(m, d, objects)

        setup_cable_harness(phy, viz)
        randomize_qpos(phy, rng, viz)

        sdf_path = root / f"{scenario.name}_{i}.sdf"

        get_sdf(sdf_path, phy, 0.01, xmin=-0.95, xmax=0.95, ymin=0.2, ymax=0.85, zmin=-0.45, zmax=1.1)

        skeletons = get_cable_harness_skeletons(phy)

        save_trial(i, phy, scenario, sdf_path, skeletons)


if __name__ == "__main__":
    main()
