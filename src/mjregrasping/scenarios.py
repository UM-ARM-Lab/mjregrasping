from dataclasses import dataclass
from pathlib import Path
from typing import Union

import mujoco
import numpy as np

from mjregrasping.physics import Physics
from mjregrasping.robot_data import RobotData, val


@dataclass
class Scenario:
    name: str
    xml_path: Path
    obstacle_name: str
    robot_data: RobotData
    rope_name: str
    noise_sigma: Union[np.ndarray, float]


real_untangle = Scenario(
    name="RealUntangle",
    xml_path=Path("models/real_scene.xml"),
    obstacle_name="obstacles",
    robot_data=val,
    rope_name="rope",
    noise_sigma=np.deg2rad(1),
)

simple_goal_sig = Scenario(
    name="SimpleGoalSig",
    xml_path=Path("models/simple_goal_sig_scene.xml"),
    obstacle_name="obstacles",
    robot_data=val,
    rope_name="rope",
    noise_sigma=np.deg2rad(1),
)

val_untangle = Scenario(
    name="Untangle",
    xml_path=Path("models/untangle_scene.xml"),
    obstacle_name="computer_rack",
    robot_data=val,
    rope_name="rope",
    noise_sigma=np.deg2rad(2),
)

threading_cable = Scenario(
    name="Threading",
    xml_path=Path("models/threading_scene.xml"),
    obstacle_name="obstacles",
    robot_data=val,
    rope_name="rope",
    noise_sigma=np.deg2rad(1),
)

val_pulling = Scenario(
    name="Pulling",
    xml_path=Path("models/pulling_scene.xml"),
    obstacle_name="obstacles",
    robot_data=val,
    rope_name="rope",
    noise_sigma=np.deg2rad(1),
)


def dx(x):
    return np.array([x, 0, 0])


def dy(y):
    return np.array([0, y, 0])


def dz(z):
    return np.array([0, 0, z])


def x_axis(d, name):
    return d.geom(name).xmat.reshape(3, 3)[:, 0]


def y_axis(d, name):
    return d.geom(name).xmat.reshape(3, 3)[:, 1]


def z_axis(d, name):
    return d.geom(name).xmat.reshape(3, 3)[:, 2]


def get_threading_skeletons(phy: Physics):
    d = phy.d
    m = phy.m
    mujoco.mj_forward(m, d)
    return {
        f"loop{k}": np.array([
            d.geom(f"loop{k}_front").xpos + dz(m.geom(f"loop{k}_front").size[2]),
            d.geom(f"loop{k}_front").xpos - dz(m.geom(f"loop{k}_front").size[2]),
            d.geom(f"loop{k}_bottom").xpos - x_axis(d, f"loop{k}_bottom") * m.geom(f"loop{k}_bottom").size[0],
            d.geom(f"loop{k}_top").xpos - x_axis(d, f"loop{k}_top") * m.geom(f"loop{k}_top").size[0],
            d.geom(f"loop{k}_front").xpos + dz(m.geom(f"loop{k}_front").size[2]),
        ]) for k in range(1, 4)
    }


def get_pulling_skeletons(phy: Physics):
    return {}


def get_untangle_skeletons(phy: Physics):
    d = phy.d
    m = phy.m
    mujoco.mj_forward(m, d)
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
    mujoco.mj_forward(m, d)
    return {
        "loop1": np.array([
            d.geom("leg1").xpos - dz(m.geom("leg1").size[2]),
            d.geom("leg1").xpos + dz(m.geom("leg1").size[2]),
            d.geom("leg4").xpos + dz(m.geom("leg4").size[2]),
            d.geom("leg4").xpos - dz(m.geom("leg4").size[2]),
            d.geom("leg1").xpos - dz(m.geom("leg1").size[2]),
        ]),
    }
