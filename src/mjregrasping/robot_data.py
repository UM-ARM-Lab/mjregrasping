from dataclasses import dataclass
from typing import List


@dataclass
class RobotData:
    # TODO: use kw_only once we move to Python 3.10
    base_link: str
    allowed_robot_collision_geoms_names: List[str]
    ignored_robot_self_collision_geoms_names: List[str]
    gripper_actuator_names: List[str]
    tool_sites: List[str]
    n_g: int
    rope_grasp_eqs: List[str]
    world_gripper_eqs: List[str]
    tool_bodies: List[str]
    ik_tol: float


val = RobotData(
    base_link="val_base",
    allowed_robot_collision_geoms_names=[
        'left_finger_pad', 'left_finger_pad2', 'right_finger_pad', 'right_finger_pad2'
    ],
    ignored_robot_self_collision_geoms_names=[
        'left_finger_pad', 'left_finger_pad2',
        'right_finger_pad', 'right_finger_pad2',
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
    tool_bodies=["drive50", "drive10"],
    ik_tol=0.01
)

conq = RobotData(
    base_link="conq_base",
    allowed_robot_collision_geoms_names=['front_left_leg', 'front_right_leg', 'back_left_leg', 'back_right_leg'],
    ignored_robot_self_collision_geoms_names=[],
    tool_sites=['hand_tool'],
    n_g=1,
    rope_grasp_eqs=['hand'],
    world_gripper_eqs=['hand_world'],
    tool_bodies=["hand"],
    gripper_actuator_names=['finger_vel'],
    ik_tol=0.02
)
