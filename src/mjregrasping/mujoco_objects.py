from copy import copy

import mujoco
import numpy as np

from mjregrasping.mujoco_object import MjObject
from mjregrasping.robot_data import RobotData


class MjObjects:

    def __init__(self, model, obstacle_name, rd: RobotData, rope_name):
        self.obstacle_name = obstacle_name
        self.rd = rd
        self.rope_name = rope_name
        self.robot = MjObject(model, rd.base_link)
        self.rope = MjObject(model, rope_name)
        self.obstacle = MjObject(model, obstacle_name)

        # allow collision between the environment and the finger pads
        self.robot_collision_geom_names = copy(self.robot.geom_names)
        for geom_name in rd.allowed_robot_collision_geoms_names:
            self.robot_collision_geom_names.remove(geom_name)

        self.robot_self_collision_geom_names = copy(self.robot.geom_names)
        for geom_name in rd.ignored_robot_self_collision_geoms_names:
            self.robot_self_collision_geom_names.remove(geom_name)


def parents_points(m: mujoco.MjModel, d: mujoco.MjData, body_name: str):
    """
    Returns the positions of a body and all its parents in order

    Args:
        m: model
        d: data
        body_name:

    Returns:
        A matrix [n, 3] of positions, where the first is the position of body_name and the last is always the root.
        The world is not included because it is always 0,0,0

    """
    body = m.body(body_name)
    body_id = body.id
    if body_id == -1:
        raise ValueError(f"body {body_name} not found")

    if body_id == 0:
        return np.zeros((1, 3))

    positions = []
    while body_id != 0:
        positions.append(d.body(body_id).xpos)
        body_id = m.body(body_id).parentid

    positions = np.array(positions)
    return positions


def parent_q_indices(m: mujoco.MjModel, body_name: str):
    """
    Returns the indices of joints of parents of a body in order from lowest to highest

    Args:
        m: model
        body_name: body name

    Returns:
        A matrix [n] of indices
    """
    body = m.body(body_name)
    body_id = body.id
    if body_id == -1:
        raise ValueError(f"body {body_name} not found")

    if body_id == 0:
        return np.zeros((1, 3))

    q_indices = []
    while True:
        q_index = m.body(body_id).jntadr[0]
        q_indices.append(q_index)
        body_id = m.body(body_id).parentid
        if q_index <= 0:
            break

    q_indices = np.sort(q_indices)
    return q_indices
