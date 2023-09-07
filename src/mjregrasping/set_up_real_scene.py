import mujoco

from mjregrasping.grasping import activate_grasp
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.physics import Physics
from mjregrasping.real_val import RealValCommander
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.val_dup import val_dedup
from mjregrasping.viz import Viz


def set_up_real_scene(val_cmd: RealValCommander, phy: Physics, viz: Viz):
    pos_in_mj_order = val_cmd.get_latest_qpos_in_mj_order()
    val_cmd.update_mujoco_qpos(phy)
    viz.viz(phy, False)
    pid_to_joint_config(phy, viz, val_dedup(pos_in_mj_order), DEFAULT_SUB_TIME_S, reached_tol=2)
    mujoco.mj_step(phy.m, phy.d, 1000)
    viz.viz(phy, False)
    activate_grasp(phy, 'right', 1)
    mujoco.mj_step(phy.m, phy.d, 250)
    viz.viz(phy, False)
    pid_to_joint_config(phy, viz, val_dedup(pos_in_mj_order), DEFAULT_SUB_TIME_S, reached_tol=2)
