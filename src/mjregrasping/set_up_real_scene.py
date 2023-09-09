import mujoco

from mjregrasping.grasping import activate_grasp
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.physics import Physics
from mjregrasping.real_val import RealValCommander
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.val_dup import val_dedup
from mjregrasping.viz import Viz


def set_up_real_scene(val_cmd: RealValCommander, phy: Physics, viz: Viz, loc: float):
    pos_in_mj_order = val_cmd.get_latest_qpos_in_mj_order()
    val_cmd.update_mujoco_qpos(phy)
    viz.viz(phy, False)
    q = val_dedup(pos_in_mj_order)
    real_gripper_q = q[-1]
    q[-1] = 0.2
    pid_to_joint_config(phy, viz, q, DEFAULT_SUB_TIME_S, reached_tol=2)
    mujoco.mj_step(phy.m, phy.d, 1000)
    viz.viz(phy, False)
    activate_grasp(phy, 'right', loc)

    # set the conaffinity of the grippers to 0 so that they don't collide with the rope,
    # let the Eq settle a bit, then set it back to 1 and let the Eq settle again.
    from itertools import chain
    for geom_name in chain(*phy.o.rd.gripper_geom_names):
        phy.m.geom(geom_name).conaffinity = 0
        phy.m.geom(geom_name).contype = 0

    for _ in range(10):
        mujoco.mj_step(phy.m, phy.d, 25)
        viz.viz(phy, False)

    for geom_name in chain(*phy.o.rd.gripper_geom_names):
        phy.m.geom(geom_name).conaffinity = 1
        phy.m.geom(geom_name).contype = 1

    q[-1] = real_gripper_q
    pid_to_joint_config(phy, viz, val_dedup(pos_in_mj_order), DEFAULT_SUB_TIME_S, reached_tol=2)
