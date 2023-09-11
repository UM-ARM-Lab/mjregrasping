from typing import Optional

import numpy as np

from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import activate_grasp, let_rope_move_through_gripper_geoms
from mjregrasping.movie import MjMovieMaker
from mjregrasping.physics import get_gripper_ctrl_indices, get_q, Physics
from mjregrasping.real_val import RealValCommander
from mjregrasping.rollout import control_step, DEFAULT_SUB_TIME_S
from mjregrasping.viz import Viz


def deactivate_moving(phy, strategy, viz: Optional[Viz], is_planning: bool, mov: Optional[MjMovieMaker] = None,
                      val_cmd: Optional[RealValCommander] = None, **kwargs):
    needs_release = [s == Strategies.MOVE for s in strategy]
    deactivate_and_settle(phy, needs_release, viz, is_planning, mov, val_cmd, **kwargs)


def deactivate_release(phy, strategy, viz: Optional[Viz], is_planning: bool, mov: Optional[MjMovieMaker] = None,
                       val_cmd: Optional[RealValCommander] = None, **kwargs):
    needs_release = [s == Strategies.RELEASE for s in strategy]
    deactivate_and_settle(phy, needs_release, viz, is_planning, mov, val_cmd, **kwargs)


def deactivate_release_and_moving(phy, strategy, viz: Optional[Viz], is_planning: bool,
                                  mov: Optional[MjMovieMaker] = None, val_cmd: Optional[RealValCommander] = None,
                                  **kwargs):
    needs_release = [s in [Strategies.MOVE, Strategies.RELEASE] for s in strategy]
    deactivate_and_settle(phy, needs_release, viz, is_planning, mov, val_cmd, **kwargs)


def deactivate_and_settle(phy, needs_release, viz: Optional[Viz], is_planning: bool,
                          mov: Optional[MjMovieMaker] = None, val_cmd: Optional[RealValCommander] = None,
                          n_open_steps: int = 5):
    rope_grasp_eqs = phy.o.rd.rope_grasp_eqs
    ctrl = np.zeros(phy.m.nu)
    gripper_ctrl_indices = get_gripper_ctrl_indices(phy)
    for eq_name, release_i, ctrl_i in zip(rope_grasp_eqs, needs_release, gripper_ctrl_indices):
        eq = phy.m.eq(eq_name)
        if release_i:
            eq.active = 0
            ctrl[ctrl_i] = 0.4
    if val_cmd:
        val_cmd.set_cdcpd_grippers(phy)
    for _ in range(n_open_steps):
        control_step(phy, ctrl, sub_time_s=DEFAULT_SUB_TIME_S, mov=mov, val_cmd=val_cmd)
    # We only want to settle, not move the robot, and because CDCPD is likely wrong when releasing,
    # we certainly don't want the mujoco rope to be constrained to stay close to CDCPD
    settle_with_checks(phy, viz, is_planning, mov, val_cmd=None)
    if val_cmd:
        # CDCPD does not do a great job when the rope moves really fast, such as when we drop it. This
        # basically resets CDCPD to what mujoco thinks the rope looks like after it has dropped
        val_cmd.set_cdcpd_from_mj_rope(phy)


def grasp_and_settle(phy, grasp_locs, viz: Optional[Viz], is_planning: bool, mov: Optional[MjMovieMaker] = None,
                     val_cmd: Optional[RealValCommander] = None, n_close_steps: int = 5):
    rope_grasp_eqs = phy.o.rd.rope_grasp_eqs
    ctrl = np.zeros(phy.m.nu)
    gripper_ctrl_indices = get_gripper_ctrl_indices(phy)
    for eq_name, grasp_loc_i, ctrl_i in zip(rope_grasp_eqs, grasp_locs, gripper_ctrl_indices):
        if grasp_loc_i == -1:
            continue
        ctrl[ctrl_i] = -0.4
        activate_grasp(phy, eq_name, grasp_loc_i)
        # let_rope_move_through_gripper_geoms(phy, 200)
    if val_cmd:
        val_cmd.set_cdcpd_grippers(phy)
    for _ in range(n_close_steps):
        control_step(phy, ctrl, sub_time_s=DEFAULT_SUB_TIME_S, mov=mov, val_cmd=val_cmd)
    settle_with_checks(phy, viz, is_planning, mov, val_cmd=None)
    if val_cmd:
        print("Resetting CDCPD to mujoco match rope state post grasp")
        val_cmd.set_cdcpd_from_mj_rope(phy)


def settle_with_checks(phy: Physics, viz: Optional[Viz], is_planning: bool, mov: Optional[MjMovieMaker] = None,
                       val_cmd: Optional[RealValCommander] = None):
    """
    In contrast to settle(), which steps for a fixed number of steps, this function steps until the rope and robot
    have settled.
    """
    ctrl = np.zeros(phy.m.nu)
    last_rope_points = get_rope_points(phy)
    last_q = get_q(phy)
    max_t = 50
    for t in range(max_t):
        control_step(phy, ctrl, sub_time_s=DEFAULT_SUB_TIME_S, mov=mov, val_cmd=val_cmd)
        rope_points = get_rope_points(phy)
        q = get_q(phy)
        if viz:
            viz.viz(phy, is_planning=is_planning)
        rope_displacements = np.linalg.norm(rope_points - last_rope_points, axis=-1)
        robot_displacements = np.abs(q - last_q)
        is_unstable = phy.d.warning.number.sum() > 0
        rope_settled = np.max(rope_displacements) < 0.01
        robot_settled = np.max(robot_displacements) < np.deg2rad(1)
        if robot_settled and rope_settled or is_unstable:
            return
        last_rope_points = rope_points
        last_q = q
    if not is_planning:
        print(f'WARNING: settle_with_checks failed to settle after {max_t} steps')
