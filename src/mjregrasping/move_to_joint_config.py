from typing import Optional

import mujoco
import numpy as np

from mjregrasping.movie import MjMovieMaker
from mjregrasping.params import hp
from mjregrasping.physics import Physics, get_q
from mjregrasping.rollout import control_step, DEFAULT_SUB_TIME_S
from mjregrasping.val_dup import val_dup
from mjregrasping.viz import Viz


def fix_qs(phy: Physics, viz: Viz, qs):
    rng = np.random.RandomState(0)
    phy_plan = phy.copy_data()
    # Since the RRT refuses to give me truly collision free plans,
    # We use the following janky planning method to work around this:
    new_qs = [qs[0]]
    for q, next_q in zip(qs[:-1], qs[1:]):
        # check if there is a collision free path between the current q and the next q
        # if there is, then we can just add the next q to the list of qs
        # if there isn't, then we need to find a collision free intermediate q
        # and then add that to the list of qs

        is_contact = check_edge_for_collision(phy_plan, next_q, q)
        print(f"{is_contact=}")
        viz.viz(phy_plan, is_planning=True)
        if not is_contact:
            new_qs.append(next_q)
        else:
            while True:
                # find a collision free intermediate q
                q_prime = sample_collision_free_waypoint(phy_plan, q, next_q, rng)
                # now check the paths between q and q_prime and q_prime and next_q
                if check_edge_for_collision(phy_plan, q_prime, q):
                    continue
                if check_edge_for_collision(phy_plan, next_q, q_prime):
                    continue
                break
            # now we know the new path is collision free. The path should be collision free up until next_q
            new_qs.append(q_prime)
            new_qs.append(next_q)

    return qs


def sample_collision_free_waypoint(phy_plan, q, next_q, rng):
    q_new = (q + next_q) / 2
    while True:
        q_prime = q_new + rng.normal(0, hp['noise_sigma'], size=q_new.shape)
        is_contact = check_q_for_collision(phy_plan, q_prime)
        if not is_contact:
            break
    return q_prime


def check_edge_for_collision(phy: Physics, next_q, q, num=10):
    q_interp = np.linspace(q, next_q, num)
    for q_interp_i in q_interp:
        is_contact = check_q_for_collision(phy, q_interp_i)
        if is_contact:
            return True
    return False


def check_q_for_collision(phy_plan, q):
    robot_geoms = phy_plan.o.robot.geom_names
    obs_geoms = phy_plan.o.obstacle.geom_names
    q_interp_i_dup = val_dup(q)
    phy_plan.d.qpos[phy_plan.o.robot.qpos_indices] = q_interp_i_dup
    mujoco.mj_forward(phy_plan.m, phy_plan.d)
    for contact in phy_plan.d.contact:
        geom_name1 = phy_plan.m.geom(contact.geom1).name
        geom_name2 = phy_plan.m.geom(contact.geom2).name
        if (geom_name1 in robot_geoms and geom_name2 in obs_geoms) or \
                (geom_name2 in robot_geoms and geom_name1 in obs_geoms):
            print(f"Contact between {geom_name1} and {geom_name2}")
            return True
    return False


def execute_grasp_plan(phy: Physics, qs, viz: Viz, is_planning: bool, mov: Optional[MjMovieMaker] = None):
    # qs = fix_qs(phy, viz, qs)

    for q in qs[:-1]:
        pid_to_joint_config(phy, viz, q, DEFAULT_SUB_TIME_S, is_planning, mov, reached_tol=2.0, stopped_tol=10.0)
    pid_to_joint_config(phy, viz, qs[-1], DEFAULT_SUB_TIME_S, is_planning, mov)


def pid_to_joint_config(phy: Physics, viz: Optional[Viz], q_target, sub_time_s, is_planning: bool = False,
                        mov: Optional[MjMovieMaker] = None, reached_tol=1.0, stopped_tol=0.5):
    q_prev = get_q(phy)
    for i in range(75):
        if viz:
            viz.viz(phy, is_planning)
        q_current = get_q(phy)
        command = hp['joint_kp'] * (q_target - q_current)

        # take the step on the real phy
        control_step(phy, command, sub_time_s=sub_time_s, mov=mov)

        # get the new current q
        q_current = get_q(phy)

        error = np.abs(q_current - q_target)
        max_joint_error = np.max(error)
        offending_q_idx = np.argmax(error)
        abs_qvel = np.abs(q_prev - q_current)
        offending_qvel_idx = np.argmax(abs_qvel)
        # NOTE: this assumes all joints are rotational...
        reached = np.rad2deg(max_joint_error) < reached_tol
        stopped = np.rad2deg(np.max(abs_qvel)) < stopped_tol
        if reached and stopped:
            return
        elif stopped and i > 10:
            break

        q_prev = q_current

    if not reached:
        name = phy.m.actuator(offending_q_idx).name
        reason = f"actuator {name} is {np.rad2deg(max_joint_error)} deg away from target."
    else:
        reason = f"qpos {offending_qvel_idx} is still moving too fast."
    print(f"PID failed to converge. {reason}")
