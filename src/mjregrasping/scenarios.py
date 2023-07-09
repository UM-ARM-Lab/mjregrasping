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
    vg_path: Path
    obstacle_name: str
    robot_data: RobotData
    rope_name: str


conq_hose = Scenario(
    name="ConqHose",
    xml_path=Path("models/conq_hose_scene.xml"),
    skeletons_path=Path("models/hose_obstacles_skeleton.hjson"),
    vg_path=Path("voxelgrids/computer_rack_vg.pkl"),
    obstacle_name="hose_obstacles",
    robot_data=conq,
    rope_name="rope",
)

val_pull = Scenario(
    name="Pull",
    xml_path=Path("models/pull_scene.xml"),
    skeletons_path=None,
    vg_path=Path("voxelgrids/pull_vg.pkl"),
    obstacle_name="floor_obstacles",
    robot_data=val,
    rope_name="rope",
)

val_untangle = Scenario(
    name="Untangle",
    xml_path=Path("models/untangle_scene.xml"),
    skeletons_path=Path("models/computer_rack_skeleton.hjson"),
    vg_path=Path("voxelgrids/computer_rack_vg.pkl"),
    obstacle_name="computer_rack",
    robot_data=val,
    rope_name="rope",
)


def setup_untangle_scene(phy, viz):
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
    activate_grasp(phy, 'right', 0.95, rope_body_indices)
    robot_q2[-1] = 0.05  # close right gripper
    pid_to_joint_config(phy, viz, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)


def make_untangle_goal(viz):
    loc = 1
    goal_point = np.array([1.5, -1.5, 0.05])
    goal = ObjectPointGoal(goal_point, 0.05, loc, viz)
    return goal


def setup_pull_scene(phy, viz):
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
    activate_grasp(phy, 'left', 0.0, rope_body_indices)
    robot_q2[9] = 0.1
    pid_to_joint_config(phy, viz, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)


def make_pull_goal(viz):
    goal_rope_loc = 1
    goal_point = np.array([0.8, 0.21, 0.2])
    goal = ObjectPointGoal(goal_point, 0.03, goal_rope_loc, viz)
    return goal


def setup_conq_hose_scene(phy, viz):
    # TODO: set up the scene so that the rope (hose) is at the attach point on the vacuum,
    #  put the hand near the end of the rope, activate the grasp constraint
    #  maybe set up the joint angles for the rope so that it is in a reasonable position.
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
    goal_point = np.array([0.78, 0.04, 1.25])
    goal = ObjectPointGoal(goal_point, goal_radius=0.05, loc=1, viz=viz)
    return goal
