import mujoco
import numpy as np

from mjregrasping.grasping import activate_grasp, let_rope_move_through_gripper_geoms
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.physics import Physics
from mjregrasping.real_val import RealValCommander
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.val_dup import val_dedup
from mjregrasping.viz import Viz


def set_up_real_scene(val_cmd: RealValCommander, phy: Physics, viz: Viz, loc: float):
    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)

    pos_in_mj_order = val_cmd.get_latest_qpos_in_mj_order()
    val_cmd.update_mujoco_qpos(phy)
    viz.viz(phy, False)
    pid_to_joint_config(phy, viz, val_dedup(pos_in_mj_order), DEFAULT_SUB_TIME_S, reached_tol=2)
    # q = np.array([
    #     0.0, -0.2,  # torso
    #     -0.9, 0.0, 0.0, 0.3, 0, 0, 0,  # left arm
    #     0.3,  # left gripper
    #     -0.9, 0.0, 0.2, 0.3, 0, 0.0, 1.5707,  # right arm
    #     0.5,  # right gripper
    # ])
    # phy.d.qpos[phy.o.robot.qpos_indices] = val_dup(q)
    # pid_to_joint_config(phy, viz, q, sub_time_s=DEFAULT_SUB_TIME_S)
    # q[-1] = -1
    # pid_to_joint_config(phy, viz, q, sub_time_s=DEFAULT_SUB_TIME_S)

    # Move the tip of the rope around to more or less match the start state in the real world
    waypoints = np.array([
        [0.6, 0.6, 0.9],
        [0.6, 0.6, 0.5],
        [0.6, 0.2, 0.5],
        [0.4, 0.2, 0.2],
    ])
    tip_eq = phy.m.eq("B_24")
    tip_eq.active = 1
    tip_eq.data[3:6] = 0
    phy.d.ctrl[:] = 0
    for waypoint in waypoints:
        tip_eq.data[0:3] = waypoint
        mujoco.mj_step(phy.m, phy.d, 1000)
        viz.viz(phy, False)
    tip_eq.active = 0
    activate_grasp(phy, 'right', loc)
    let_rope_move_through_gripper_geoms(phy, 200)
    pid_to_joint_config(phy, viz, val_dedup(pos_in_mj_order), DEFAULT_SUB_TIME_S, reached_tol=2)

