from time import sleep

import mujoco
import numpy as np

from mjregrasping.grasping import activate_grasp, let_rope_move_through_gripper_geoms, disable_rope_gripper_collision
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.my_transforms import wxyz_quat_from_euler
from mjregrasping.physics import Physics
from mjregrasping.real_val import RealValCommander
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.val_dup import val_dedup
from mjregrasping.viz import Viz


def set_up_real_scene(val_cmd: RealValCommander, phy: Physics, viz: Viz, loc: float):
    # disable collision only between rope and gripper geoms
    from itertools import chain
    for geom_name in list(chain(*phy.o.rd.gripper_geom_names)) + phy.o.rope.geom_names:
        phy.m.geom(geom_name).contype = 1
        phy.m.geom(geom_name).conaffinity = 2

    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)

    pos_in_mj_order = val_cmd.get_latest_qpos_in_mj_order()
    val_cmd.update_mujoco_qpos(phy)
    viz.viz(phy, False)
    pid_to_joint_config(phy, viz, val_dedup(pos_in_mj_order), DEFAULT_SUB_TIME_S)

    # Move the tip of the rope a long a predefined path
    waypoints = np.array([
        [0.6, 0, 0.1],
        [0.6, 0.7, 0.4],
        [0.2, 0.7, 0.4],
    ])
    tip_eq = phy.m.eq("B_24")
    tip_eq.active = 1
    tip_eq.data[3:6] = 0
    for waypoint in waypoints:
        tip_eq.data[0:3] = waypoint
        mujoco.mj_step(phy.m, phy.d, 500)
        viz.viz(phy)

    tip_eq.active = 0

    activate_grasp(phy, 'right', loc)
    phy.m.eq("right").data[6:10] = wxyz_quat_from_euler(1.0, 0, 0)

    phy.d.ctrl[:] = 0
    for _ in range(50):
        mujoco.mj_step(phy.m, phy.d, 20)
        viz.viz(phy)

    pid_to_joint_config(phy, viz, val_dedup(pos_in_mj_order), DEFAULT_SUB_TIME_S)

    val_cmd.set_cdcpd_grippers(phy)
    # Try to let CDCPD and mujoco agree
    for _ in range(3):
        val_cmd.set_cdcpd_from_mj_rope(phy)
        sleep(2)
        val_cmd.pull_rope_towards_cdcpd(phy, 100)
        mujoco.mj_step(phy.m, phy.d, 100)
        viz.viz(phy)
    viz.viz(phy)
