from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class RobotData:
    # TODO: use kw_only once we move to Python 3.10
    base_link: str
    allowed_robot_collision_geoms_names: List[str]
    ignored_robot_self_collision_geoms_names: List[str]
    gripper_geom_names: List[List[str]]
    gripper_actuator_names: List[str]
    gripper_joint_names: List[str]
    tool_sites: List[str]
    camera_names: List[str]
    n_g: int
    rope_grasp_eqs: List[str]
    world_gripper_eqs: List[str]
    world_rope_eqs: List[str]
    tool_bodies: List[str]
    ik_tol: float
    noise_sigma: np.ndarray
    workspace_bbox: np.ndarray
    q_home: np.ndarray


val = RobotData(
    base_link="val_base",
    allowed_robot_collision_geoms_names=[
        'left_finger_pad', 'left_finger_pad2', 'right_finger_pad', 'right_finger_pad2'
    ],
    ignored_robot_self_collision_geoms_names=[
        'left_finger_pad', 'left_finger_pad2',
        'right_finger_pad', 'right_finger_pad2',
    ],
    gripper_geom_names=[
        ['left_finger_pad', 'left_finger_pad2', 'leftgripper', 'leftgripper2'],
        ['right_finger_pad', 'right_finger_pad2', 'rightgripper', 'rightgripper2'],
    ],
    gripper_joint_names=[
        'leftgripper', 'leftgripper2',
        'rightgripper', 'rightgripper2'
    ],
    gripper_actuator_names=['leftgripper_vel', 'rightgripper_vel'],
    tool_sites=[
        'left_tool', 'right_tool'
    ],
    n_g=2,
    rope_grasp_eqs=[
        'left', 'right'
    ],
    world_gripper_eqs=[
        'left_world', 'right_world'
    ],
    world_rope_eqs=[
        'rope_left_world', 'rope_right_world'
    ],
    tool_bodies=["drive50", "drive10"],
    camera_names=['left_hand', 'right_hand'],
    ik_tol=0.01,
    noise_sigma=np.deg2rad(2),
    workspace_bbox=np.array([
        [0.7, 1.2],
        [-0.6, 0.6],
        [0.5, 1.5]
    ]),
    q_home=np.array([
        0, 0,
        0, 0.15, 0, 0, 0, 0, 0,
        0,
        0, -0.15, 0, 0, 0, 0, 0,
        0
    ])
)

conq = RobotData(
    base_link="conq_base",
    allowed_robot_collision_geoms_names=['front_left_leg', 'front_right_leg', 'back_left_leg', 'back_right_leg'],
    ignored_robot_self_collision_geoms_names=[],
    gripper_geom_names=[['hand']],
    tool_sites=['hand_tool'],
    n_g=1,
    rope_grasp_eqs=['hand'],
    world_gripper_eqs=['hand_world'],
    world_rope_eqs=['rope_hand_world'],
    tool_bodies=["hand"],
    gripper_actuator_names=['finger_vel'],
    gripper_joint_names=['finger'],
    ik_tol=0.02,
    camera_names=['hand'],
    noise_sigma=np.array([0.02, 0.02, 0.01, np.deg2rad(1)]),
    workspace_bbox=np.array([
        [0, 1.5],
        [-1.0, 1.0],
        [0.0, 0.5],
    ]),
    q_home=np.array([])
)
