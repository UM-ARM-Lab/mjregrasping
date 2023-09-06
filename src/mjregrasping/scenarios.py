from dataclasses import dataclass
from pathlib import Path
from typing import Union

import mujoco
import numpy as np
from transformations import quaternion_from_euler

from mjregrasping.goals import ObjectPointGoal
from mjregrasping.grasping import activate_grasp
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.physics import Physics
from mjregrasping.robot_data import RobotData, conq, val
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.settle import settle


@dataclass
class Scenario:
    name: str
    xml_path: Path
    obstacle_name: str
    robot_data: RobotData
    rope_name: str
    noise_sigma: Union[np.ndarray, float]


conq_hose = Scenario(
    name="ConqHose",
    xml_path=Path("models/conq_hose_scene.xml"),
    obstacle_name="hose_obstacles",
    robot_data=conq,
    rope_name="rope",
    noise_sigma=np.array([0.02, 0.02, 0.01, np.deg2rad(1)]),
)

real_untangle = Scenario(
    name="RealUntangle",
    xml_path=Path("models/real_scene.xml"),
    obstacle_name="obstacles",
    robot_data=val,
    rope_name="rope",
    noise_sigma=np.deg2rad(2),
)

val_untangle = Scenario(
    name="Untangle",
    xml_path=Path("models/untangle_scene.xml"),
    obstacle_name="computer_rack",
    robot_data=val,
    rope_name="rope",
    noise_sigma=np.deg2rad(2),
)

threading = Scenario(
    name="Threading",
    xml_path=Path("models/threading_scene.xml"),
    obstacle_name="obstacles",
    robot_data=val,
    rope_name="rope",
    noise_sigma=np.deg2rad(1),
)


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


def dx(x):
    return np.array([x, 0, 0])


def dy(y):
    return np.array([0, y, 0])


def dz(z):
    return np.array([0, 0, z])


def get_threading_skeletons(phy: Physics):
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


def get_untangle_skeletons(phy: Physics):
    d = phy.d
    m = phy.m
    return {
        "loop1": d.geom("rack1_post1").xpos - dz(m.geom("rack1_post1").size[2]) + np.cumsum([
            np.zeros(3),
            dy(m.geom("rack2_bottom").size[0]) * 2,
            dz(m.geom("rack1_post1").size[2] * 2),
            -dy(m.geom("rack2_bottom").size[0] * 2),
            dz(-m.geom("rack1_post1").size[2] * 2),
        ], axis=0),
        "loop2": d.geom("rack1_post2").xpos - dz(m.geom("rack1_post2").size[2]) + np.cumsum([
            np.zeros(3),
            dy(m.geom("rack2_bottom").size[0]) * 2,
            dz(m.geom("rack1_post2").size[2] * 2),
            -dy(m.geom("rack2_bottom").size[0] * 2),
            dz(-m.geom("rack1_post2").size[2] * 2),
        ], axis=0),
        "loop3": d.geom("rack1_post1").xpos - dz(m.geom("rack1_post1").size[2]) + np.cumsum([
            np.zeros(3),
            -dx(m.geom("rack2_bottom").size[1]) * 2,
            dz(m.geom("rack1_post1").size[2] * 2),
            dx(m.geom("rack2_bottom").size[1] * 2),
            dz(-m.geom("rack1_post1").size[2] * 2),
        ], axis=0),
        "loop4": d.geom("rack1_post3").xpos - dz(m.geom("rack1_post3").size[2]) + np.cumsum([
            np.zeros(3),
            -dx(m.geom("rack2_bottom").size[1]) * 2,
            dz(m.geom("rack1_post3").size[2] * 2),
            dx(m.geom("rack2_bottom").size[1] * 2),
            dz(-m.geom("rack1_post3").size[2] * 2),
        ], axis=0),
        # now again for rack2
        "loop5": d.geom("rack2_post1").xpos - dz(m.geom("rack2_post1").size[2]) + np.cumsum([
            np.zeros(3),
            dy(m.geom("rack2_bottom").size[0]) * 2,
            dz(m.geom("rack2_post1").size[2] * 2),
            -dy(m.geom("rack2_bottom").size[0] * 2),
            dz(-m.geom("rack2_post1").size[2] * 2),
        ], axis=0),
        "loop6": d.geom("rack2_post2").xpos - dz(m.geom("rack2_post2").size[2]) + np.cumsum([
            np.zeros(3),
            dy(m.geom("rack2_bottom").size[0]) * 2,
            dz(m.geom("rack2_post2").size[2] * 2),
            -dy(m.geom("rack2_bottom").size[0] * 2),
            dz(-m.geom("rack2_post2").size[2] * 2),
        ], axis=0),
        "loop7": d.geom("rack2_post1").xpos - dz(m.geom("rack2_post1").size[2]) + np.cumsum([
            np.zeros(3),
            -dx(m.geom("rack2_bottom").size[1]) * 2,
            dz(m.geom("rack2_post1").size[2] * 2),
            dx(m.geom("rack2_bottom").size[1] * 2),
            dz(-m.geom("rack2_post1").size[2] * 2),
        ], axis=0),
        "loop8": d.geom("rack2_post3").xpos - dz(m.geom("rack2_post3").size[2]) + np.cumsum([
            np.zeros(3),
            -dx(m.geom("rack2_bottom").size[1]) * 2,
            dz(m.geom("rack2_post3").size[2] * 2),
            dx(m.geom("rack2_bottom").size[1] * 2),
            dz(-m.geom("rack2_post3").size[2] * 2),
        ], axis=0),
    }


def get_real_untangle_skeletons(phy: Physics):
    d = phy.d
    m = phy.m
    return {
        "loop1": np.array([
            d.geom("leg1").xpos - dz(m.geom("leg1").size[2]),
            d.geom("leg1").xpos + dz(m.geom("leg1").size[2]),
            d.geom("leg4").xpos + dz(m.geom("leg4").size[2]),
            d.geom("leg4").xpos - dz(m.geom("leg4").size[2]),
            d.geom("leg1").xpos - dz(m.geom("leg1").size[2]),
        ]),
    }
