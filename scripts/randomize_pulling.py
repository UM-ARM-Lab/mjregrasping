#!/usr/bin/env python3
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import rerun as rr

from arc_utilities import ros_init
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import activate_grasp
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.physics import Physics, get_q
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.rrt import GraspRRT
from mjregrasping.rviz import MujocoXmlExpander
from mjregrasping.scenarios import val_untangle, get_untangle_skeletons, val_pulling, get_pulling_skeletons
from mjregrasping.teleport_to_plan import teleport_to_end_of_plan
from mjregrasping.trials import save_trial
from mjregrasping.viz import make_viz, Viz
from moveit_msgs.msg import MoveItErrorCodes


def randomize(original_path: Path, rng: np.random.RandomState):
    mxml = MujocoXmlExpander(original_path)

    attach = mxml.get_e("body", "attach")
    attach_pos = mxml.get_vec(attach, 'pos')
    attach_pos[1] += rng.uniform(-0.05, 0.05)
    attach_pos[2] += rng.uniform(-0.05, 0.05)
    mxml.set_vec(attach, attach_pos, 'pos')

    wall = mxml.get_e("body", "wall")
    wall_pos = mxml.get_vec(wall, 'pos')
    wall_pos[0] += rng.uniform(-0.05, 0.05)
    mxml.set_vec(wall, wall_pos, 'pos')

    tmp_path = mxml.save_tmp()
    m = mujoco.MjModel.from_xml_path(str(tmp_path))
    return m


def randomize_qpos(phy: Physics, rng: np.random.RandomState, viz: Optional[Viz]):
    q = get_q(phy)
    q[0] += np.deg2rad(rng.uniform(-5, 5))
    q[1] += np.deg2rad(rng.uniform(-5, 5))
    pid_to_joint_config(phy, viz, q, sub_time_s=DEFAULT_SUB_TIME_S)


@ros_init.with_ros("randomize")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('randomize')
    rr.connect()

    scenario = val_pulling

    viz = make_viz(scenario)

    root = Path("trial_data") / scenario.name
    root.mkdir(exist_ok=True, parents=True)

    rng = np.random.RandomState(0)
    for trial_idx in range(0, 25):
        # Configure the model before we construct the data and physics object
        m = randomize(scenario.xml_path, rng)

        d = mujoco.MjData(m)
        objects = MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
        phy = Physics(m, d, objects)
        # Must call mj_forward or the skeletons will be wrong
        mujoco.mj_forward(m, d)
        viz.viz(phy)

        skeletons = get_pulling_skeletons(phy)
        viz.skeletons(skeletons)

        # randomly shove the rope around
        eq = phy.m.eq("perturb")
        eq.active = 1
        workspace_min = np.array([-0.25, 0.3, 0.0])
        workspace_max = np.array([1.5, 1.5, 0.3])
        eq.data[0:3] = (workspace_max + workspace_min) / 2
        for loc in rng.uniform(0, 1, 6):
            activate_grasp(phy, 'perturb', loc)
            dx, dy = rng.normal(0, 0.3, 2)
            eq.data[0:3] += np.array([dx, dy, 0])
            # clip to workspace
            eq.data[0:3] = np.clip(eq.data[0:3], workspace_min, workspace_max)
            for t in range(10):
                mujoco.mj_step(phy.m, phy.d, 10)
                viz.viz(phy)

        eq.active = 0
        for t in range(50):
            mujoco.mj_step(phy.m, phy.d, 10)
            viz.viz(phy)

        save_trial(trial_idx, phy, scenario, None, skeletons)
        print(f"Trial {trial_idx} saved")


if __name__ == "__main__":
    main()
