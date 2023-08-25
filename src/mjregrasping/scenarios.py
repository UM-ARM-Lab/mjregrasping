from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import mujoco
import numpy as np
from transformations import quaternion_from_euler

from mjregrasping.goals import ObjectPointGoal, ThreadingGoal, GraspLocsGoal
from mjregrasping.grasping import activate_grasp
from mjregrasping.homotopy_utils import load_skeletons
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.physics import Physics
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
    noise_sigma: Union[np.ndarray, float]


conq_hose = Scenario(
    name="ConqHose",
    xml_path=Path("models/conq_hose_scene.xml"),
    skeletons_path=Path("models/hose_obstacles_skeleton.hjson"),
    sdf_path=Path("sdfs/hose_obstacles.sdf"),
    obstacle_name="hose_obstacles",
    robot_data=conq,
    rope_name="rope",
    noise_sigma=np.array([0.02, 0.02, 0.01, np.deg2rad(1)]),
)

val_untangle = Scenario(
    name="Untangle",
    xml_path=Path("models/untangle_scene.xml"),
    skeletons_path=Path("models/computer_rack_skeleton.hjson"),
    sdf_path=Path("sdfs/computer_rack.sdf"),
    obstacle_name="computer_rack",
    robot_data=val,
    rope_name="rope",
    noise_sigma=np.deg2rad(2)
)

cable_harness = Scenario(
    name="CableHarness",
    xml_path=Path("models/cable_harness_scene.xml"),
    skeletons_path=Path("models/cable_harness_skeleton.hjson"),
    sdf_path=Path("sdfs/cable_harness_obstacles.sdf"),
    obstacle_name="cable_harness_obstacles",
    robot_data=val,
    rope_name="rope",
    noise_sigma=np.deg2rad(1)
)


def setup_untangle(phy, viz):
    rope_xyz_q_indices = phy.o.rope.qpos_indices[:3]
    rope_quat_q_indices = phy.o.rope.qpos_indices[3:7]
    mujoco.mj_forward(phy.m, phy.d)
    phy.d.qpos[rope_xyz_q_indices] = phy.d.body("attach").xpos
    phy.d.qpos[rope_quat_q_indices] = quaternion_from_euler(0, 0, 0)
    # update the attach constraint
    phy.m.eq("attach").data[3:6] = 0
    mujoco.mj_forward(phy.m, phy.d)

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
        1.2, -0.2, 0, -0.90, 0, -0.6, 1.5707,  # right arm
        0.3,  # right gripper
    ])
    pid_to_joint_config(phy, viz, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)

    activate_grasp(phy, 'right', 1.0)
    robot_q2[-1] = 0.05  # close right gripper
    pid_to_joint_config(phy, viz, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)


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
    phy.d.qpos[rope_xyz_q_indices] = np.array([1.5, 0.6, 0.0])
    phy.d.qpos[rope_quat_q_indices] = quaternion_from_euler(0, 0.2, np.pi)

    activate_grasp(phy, 'attach1', 0.0)

    q = np.array([
        0.4, 0.8,  # torso
        0.0, 0.0, 0.0, 0.0, 0, 0, 0,  # left arm
        0,  # left gripper
        0.0, 0.0, 0, 0.0, 0, -0.0, -0.5,  # right arm
        0.2,  # right gripper
    ])
    pid_to_joint_config(phy, viz, q, sub_time_s=DEFAULT_SUB_TIME_S)
    activate_grasp(phy, 'right', 0.93)
    settle(phy, DEFAULT_SUB_TIME_S, viz, is_planning=False)
    q = np.array([
        -0.6, 0.1,  # torso
        -0.2, 0.0, 0.0, 0.0, 0, 0, 0,  # left arm
        0,  # left gripper
        -0.1, 0.0, 0, 0.0, 0, -0.0, -0.5,  # right arm
        0.06,  # right gripper
    ])
    pid_to_joint_config(phy, viz, q, sub_time_s=DEFAULT_SUB_TIME_S)
    q = np.array([
        -0.68, 0.24,  # torso
        -0.4, 0.0, 0, 0.4, 0.8, 0.2, 3.14159,  # left arm
        0,  # left gripper
        0.4, 0.0, 0, -0.13, 0, -0.0, -0.5,  # right arm
        0.06,  # right gripper
    ])
    pid_to_joint_config(phy, viz, q, sub_time_s=DEFAULT_SUB_TIME_S)


def dx(x):
    return np.array([x, 0, 0])


def dy(y):
    return np.array([0, y, 0])


def dz(z):
    return np.array([0, 0, z])


def get_cable_harness_skeletons(phy: Physics):
    d = phy.d
    m = phy.m
    return {
        "loop1": d.geom("loop1_front").xpos - dz(m.geom("loop1_front").size[2]) + np.cumsum([
            np.zeros(3),
            dy(m.geom("loop1_top").size[0]) * 2,
            dz(m.geom("loop1_front").size[2] * 2),
            -dy(m.geom("loop1_top").size[0] * 2),
            dz(-m.geom("loop1_front").size[2] * 2),
        ], axis=0),
        "loop2": d.geom("loop2_front").xpos - dz(m.geom("loop2_front").size[2]) + np.cumsum([
            np.zeros(3),
            dy(m.geom("loop2_top").size[0]) * 2,
            dz(m.geom("loop2_front").size[2] * 2),
            -dy(m.geom("loop2_top").size[0] * 2),
            dz(-m.geom("loop2_front").size[2] * 2),
        ], axis=0),
        "loop3": d.geom("loop3_front").xpos - dz(m.geom("loop3_front").size[2]) + np.cumsum([
            np.zeros(3),
            dy(m.geom("loop3_top").size[0]) * 2,
            dz(m.geom("loop3_front").size[2] * 2),
            -dy(m.geom("loop3_top").size[0] * 2),
            dz(-m.geom("loop3_front").size[2] * 2),
        ], axis=0),
    }
