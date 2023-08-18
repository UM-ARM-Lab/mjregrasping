import mujoco
import numpy as np


def teleport_to_end_of_plan(phy_plan, res):
    plan_final_q = np.array(res.trajectory.joint_trajectory.points[-1].positions)
    teleport_to_planned_q(phy_plan, plan_final_q)


def teleport_to_planned_q(phy_plan, plan_final_q):
    qpos_for_act = phy_plan.m.actuator_trnid[:, 0]
    phy_plan.d.qpos[qpos_for_act] = plan_final_q
    phy_plan.d.act = plan_final_q
    mujoco.mj_forward(phy_plan.m, phy_plan.d)
