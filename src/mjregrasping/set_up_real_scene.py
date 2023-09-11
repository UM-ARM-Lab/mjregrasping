import mujoco

from mjregrasping.grasping import activate_grasp, let_rope_move_through_gripper_geoms
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.my_transforms import wxyz_quat_from_euler
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
    pid_to_joint_config(phy, viz, val_dedup(pos_in_mj_order), DEFAULT_SUB_TIME_S)

    val_cmd.pull_rope_towards_cdcpd(phy, 500)
    viz.viz(phy)

    activate_grasp(phy, 'right', loc)
    phy.m.eq("right").data[6:10] = wxyz_quat_from_euler(0, 0, 0)

    phy.d.ctrl[:] = 0
    for _ in range(100):
        let_rope_move_through_gripper_geoms(phy, 10)
        viz.viz(phy)
    for _ in range(20):
        mujoco.mj_step(phy.m, phy.d, 10)
        viz.viz(phy)

    pid_to_joint_config(phy, viz, val_dedup(pos_in_mj_order), DEFAULT_SUB_TIME_S)
