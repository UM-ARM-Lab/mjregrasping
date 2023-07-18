from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
from transformations import quaternion_from_euler

from mjregrasping.goals import ObjectPointGoal
from mjregrasping.grasping import activate_grasp
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.robot_data import RobotData, conq, val
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.settle import settle


@dataclass
class Scenario:
    name: str
    xml_path: Path
    skeletons_path: Optional[Path]
    sdf_path: Path
    obstacle_name: str
    robot_data: RobotData
    rope_name: str


conq_hose = Scenario(
    name="ConqHose",
    xml_path=Path("models/conq_hose_scene.xml"),
    skeletons_path=Path("models/hose_obstacles_skeleton.hjson"),
    sdf_path=Path("sdfs/hose_obstacles.sdf"),
    obstacle_name="hose_obstacles",
    robot_data=conq,
    rope_name="rope",
)

val_pull = Scenario(
    name="Pull",
    xml_path=Path("models/pull_scene.xml"),
    skeletons_path=None,
    sdf_path=Path("sdfs/pull.sdf"),
    obstacle_name="floor_obstacles",
    robot_data=val,
    rope_name="rope",
)

val_untangle = Scenario(
    name="Untangle",
    xml_path=Path("models/untangle_scene.xml"),
    skeletons_path=Path("models/computer_rack_skeleton.hjson"),
    sdf_path=Path("sdfs/computer_rack.sdf"),
    obstacle_name="computer_rack",
    robot_data=val,
    rope_name="rope",
)

cable_harness = Scenario(
    name="CableHarness",
    xml_path=Path("models/cable_harness_scene.xml"),
    skeletons_path=Path("models/cable_harness_skeleton.hjson"),
    sdf_path=Path("sdfs/cable_harness_obstacles.sdf"),
    obstacle_name="cable_harness_obstacles",
    robot_data=val,
    rope_name="rope",
)


def setup_untangle(phy, viz):
    robot_q1 = np.array([
        -0.7, 0.1,  # torso
        -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
        0,  # left gripper
        0.0, -0.2, 0, -0.30, 0, -0.2, 0,  # right arm
        0.3,  # right gripper
    ])
    pid_to_joint_config(phy, viz, robot_q1, sub_time_s=DEFAULT_SUB_TIME_S)
    robot_q2 = np.array([
        -0.5, 0.4,  # torso
        -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
        0.1,  # left gripper
        1.2, -0.2, 0, -0.90, 0, -0.6, 0,  # right arm
        0.3,  # right gripper
    ])
    pid_to_joint_config(phy, viz, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)
    rope_body_indices = np.array(phy.o.rope.body_indices)
    activate_grasp(phy, 'right', 1.0)
    robot_q2[-1] = 0.05  # close right gripper
    pid_to_joint_config(phy, viz, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)


def make_untangle_goal(viz):
    loc = 1
    goal_point = np.array([0.8, 0.04, 1.25])
    goal = ObjectPointGoal(goal_point, 0.06, loc, viz)
    return goal


def setup_pull(phy, viz):
    # set the rope pose
    phy.d.qpos[20:23] = np.array([1.0, 0.2, 0])
    phy.d.qpos[23:27] = quaternion_from_euler(0, 0, 0)
    robot_q2 = np.array([
        0.0, 0.9,  # torso
        1.0, 0.0, 0.0, -1.0, 0, 0, 0,  # left arm
        0.3,  # left gripper
        1.0, 0.0, 0, 0.0, 0, 0.0, 0,  # right arm
        0.1,  # right gripper
    ])
    pid_to_joint_config(phy, viz, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)
    rope_body_indices = np.array(phy.o.rope.body_indices)
    activate_grasp(phy, 'left', 0.0)
    robot_q2[9] = 0.1
    pid_to_joint_config(phy, viz, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)


def make_pull_goal(viz):
    goal_rope_loc = 1
    goal_point = np.array([0.8, 0.21, 0.2])
    goal = ObjectPointGoal(goal_point, 0.03, goal_rope_loc, viz)
    return goal


def setup_conq_hose(phy, viz):
    rope_xyz_q_indices = phy.o.rope.qpos_indices[:3]
    rope_quat_q_indices = phy.o.rope.qpos_indices[3:7]
    rope_ball_q_indices = phy.o.rope.qpos_indices[7:].reshape((-1, 4))
    phy.d.qpos[rope_xyz_q_indices] = np.array([2.0, 1.33, 0.25])
    phy.d.qpos[rope_quat_q_indices] = quaternion_from_euler(0, 0, np.pi)
    for j in range(rope_ball_q_indices.shape[0]):
        if 2 < j < 8:
            phy.d.qpos[rope_ball_q_indices[j]] = quaternion_from_euler(0, 0, np.deg2rad(18))
        else:
            phy.d.qpos[rope_ball_q_indices[j]] = quaternion_from_euler(0, 0, 0)
    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)
    attach_eq = phy.m.eq('attach')
    attach_eq.data[3:6] = 0
    attach_eq.active = True
    settle(phy, DEFAULT_SUB_TIME_S, viz, is_planning=False)


def make_conq_hose_goal(viz):
    goal_point = np.array([1.5, -0.75, 0.04])
    goal = ObjectPointGoal(goal_point, goal_radius=0.15, loc=1, viz=viz)
    return goal


def setup_cable_harness(phy, viz):
    rope_xyz_q_indices = phy.o.rope.qpos_indices[:3]
    rope_quat_q_indices = phy.o.rope.qpos_indices[3:7]
    phy.d.qpos[rope_xyz_q_indices] = np.array([1.2, -1.0, 1.0])
    phy.d.qpos[rope_quat_q_indices] = quaternion_from_euler(0, 0, np.pi / 2)

    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)

    activate_grasp(phy, 'attach1', 0.0)
    activate_grasp(phy, 'attach2', 1.0)
    settle(phy, DEFAULT_SUB_TIME_S, viz, is_planning=False)
