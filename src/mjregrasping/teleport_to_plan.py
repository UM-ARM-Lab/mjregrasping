from typing import Optional

import mujoco
import numpy as np

from mjregrasping.movie import MjMovieMaker
from mjregrasping.physics import Physics, get_qpos_ids_for_actuators
from mjregrasping.viz import Viz


def teleport_to_end_of_plan(phy_plan, res):
    plan_final_q = np.array(res.trajectory.joint_trajectory.points[-1].positions)
    teleport_to_planned_q(phy_plan, plan_final_q)


def teleport_to_planned_q(phy_plan, plan_final_q):
    qpos_ids = get_qpos_ids_for_actuators(phy_plan)
    phy_plan.d.qpos[qpos_ids] = plan_final_q
    phy_plan.d.act = plan_final_q
    mujoco.mj_forward(phy_plan.m, phy_plan.d)


def teleport_along_plan(phy: Physics, qs, viz: Viz, is_planning: bool, mov: Optional[MjMovieMaker] = None):
    for q in qs:
        teleport_to_planned_q(phy, q)
        viz.viz(phy, is_planning)
        if mov:
            mov.render(phy.d)
