from typing import Optional

import numpy as np

from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import activate_grasp
from mjregrasping.movie import MjMovieMaker
from mjregrasping.physics import get_gripper_ctrl_indices, get_q, Physics
from mjregrasping.rollout import control_step, DEFAULT_SUB_TIME_S
from mjregrasping.settle import settle
from mjregrasping.viz import Viz


def release_and_settle(phy, strategy, viz: Optional[Viz], is_planning: bool, mov: Optional[MjMovieMaker] = None):
    needs_release = [s in [Strategies.MOVE, Strategies.RELEASE] for s in strategy]

    rope_grasp_eqs = phy.o.rd.rope_grasp_eqs
    ctrl = np.zeros(phy.m.nu)
    gripper_ctrl_indices = get_gripper_ctrl_indices(phy)
    for eq_name, release_i, ctrl_i in zip(rope_grasp_eqs, needs_release, gripper_ctrl_indices):
        eq = phy.m.eq(eq_name)
        if release_i:
            eq.active = 0
            ctrl[ctrl_i] = 0.4
    control_step(phy, ctrl, sub_time_s=DEFAULT_SUB_TIME_S * 10)
    settle_with_checks(phy, viz, is_planning, mov)


def grasp_and_settle(phy, grasp_locs, viz: Optional[Viz], is_planning: bool, mov: Optional[MjMovieMaker] = None):
    rope_grasp_eqs = phy.o.rd.rope_grasp_eqs
    ctrl = np.zeros(phy.m.nu)
    gripper_ctrl_indices = get_gripper_ctrl_indices(phy)
    for eq_name, grasp_loc_i, ctrl_i in zip(rope_grasp_eqs, grasp_locs, gripper_ctrl_indices):
        if grasp_loc_i == -1:
            continue
        ctrl[ctrl_i] = -0.4
        activate_grasp(phy, eq_name, grasp_loc_i)
    control_step(phy, ctrl, sub_time_s=DEFAULT_SUB_TIME_S * 5)
    settle_with_checks(phy, viz, is_planning, mov)


def settle_with_checks(phy: Physics, viz: Optional[Viz], is_planning: bool, mov: Optional[MjMovieMaker] = None):
    """
    In contrast to settle(), which steps for a fixed number of steps, this function steps until the rope and robot
    have settled.
    """
    ctrl = np.zeros(phy.m.nu)
    last_rope_points = get_rope_points(phy)
    last_q = get_q(phy)
    max_t = 40
    for t in range(max_t):
        control_step(phy, ctrl, sub_time_s=5 * DEFAULT_SUB_TIME_S)
        rope_points = get_rope_points(phy)
        q = get_q(phy)
        if viz:
            viz.viz(phy, is_planning=is_planning)
        if mov:
            mov.render(phy.d)
        rope_displacements = np.linalg.norm(rope_points - last_rope_points, axis=-1)
        robot_displacements = np.abs(q - last_q)
        is_unstable = phy.d.warning.number.sum() > 0
        if np.max(rope_displacements) < 0.01 and np.max(robot_displacements) < np.deg2rad(1) or is_unstable:
            return
        last_rope_points = rope_points
        last_q = q
    if not is_planning:
        print(f'WARNING: settle_with_checks failed to settle after {max_t} steps')
