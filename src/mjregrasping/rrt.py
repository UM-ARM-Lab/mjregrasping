from typing import Optional

import mujoco
import numpy as np
from pymjregrasping_cpp import RRTPlanner

import dynamic_reconfigure
import rospy
from dynamic_reconfigure.client import Client
from mjregrasping.grasp_conversions import grasp_locations_to_xpos
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_is_grasping
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.moveit_planning import make_planning_scene
from mjregrasping.params import hp
from mjregrasping.physics import Physics, get_full_q
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.val_dup import val_dedup
from mjregrasping.viz import Viz
from moveit_msgs.msg import MotionPlanResponse


class GraspRRT:

    def __init__(self):
        self.rrt = RRTPlanner()
        # For some reason, I can fix the RRT collision issues by setting these params
        client = Client("/ompl")
        client.update_configuration({"maximum_waypoint_distance": 0.15})
        self.fix_start_rng = np.random.RandomState(0)

    def plan(self, phy: Physics, strategy, locs: np.ndarray, viz: Optional[Viz], allowed_planning_time=5.0,
             pos_noise=0.02, **kwargs):
        phy_plan = phy.copy_all()
        goals, group_name, q0 = plan_to_grasp(locs, phy_plan, strategy)

        scene_msg = make_planning_scene(phy_plan)
        res: MotionPlanResponse = self.rrt.plan(scene_msg, group_name, goals, bool(viz), allowed_planning_time,
                                                pos_noise, **kwargs)
        return res, scene_msg

    def display_result(self, viz, res, scene_msg):
        viz.rviz.viz_scene(scene_msg)
        self.rrt.display_result(res)

    def is_state_valid(self, phy: Physics):
        scene_msg = make_planning_scene(phy)
        return self.rrt.is_state_valid(scene_msg)

    def fix_start_state_in_place(self, phy: Physics, viz: Optional[Viz] = None):
        for _ in range(10):
            valid = self.is_state_valid(phy)
            if valid:
                return True
            q_new, is_fixed = self.get_fixed_qpos(phy, viz)
            if is_fixed:
                pid_to_joint_config(phy, viz, val_dedup(q_new), DEFAULT_SUB_TIME_S, is_planning=False)
        return False

    def get_fixed_qpos(self, phy: Physics, viz: Optional[Viz] = None):
        """ Sample a new qpos for the robot that is collision free according to the RRT planner """
        phy_plan = phy.copy_all()
        q0 = phy_plan.d.qpos[phy_plan.o.robot.qpos_indices]
        is_fixed = False
        for _ in range(100):
            mujoco.mj_forward(phy_plan.m, phy_plan.d)
            valid = self.is_state_valid(phy_plan)
            if viz:
                viz.viz(phy_plan, is_planning=True)
                # print(valid)
            if valid:
                is_fixed = True
                break
            phy_plan.d.qpos[phy_plan.o.robot.qpos_indices] = q0 + np.deg2rad(
                self.fix_start_rng.uniform(-hp['start_state_jiggle_deg'], hp['start_state_jiggle_deg'],
                                           size=len(phy_plan.o.robot.qpos_indices)))
        return phy_plan.d.qpos[phy_plan.o.robot.qpos_indices], is_fixed


class NoArmsMoving(Exception):
    pass


def plan_to_grasp(candidate_locs, phy_plan, strategy):
    # Run RRT to find a collision free path from the current q to candidate_pos
    # Then in theory we should do it for the subgoals too
    candidate_pos = grasp_locations_to_xpos(phy_plan, candidate_locs)
    q0 = get_full_q(phy_plan)
    goals = {}
    is_grasping0 = get_is_grasping(phy_plan)
    in_planning = []
    for tool_name, s, is_grasping0_i, p in zip(phy_plan.o.rd.tool_sites, strategy, is_grasping0, candidate_pos):
        if s in [Strategies.NEW_GRASP, Strategies.MOVE]:
            goals[tool_name] = p
            in_planning.append(True)
        else:
            in_planning.append(False)
    if all(in_planning) or not any(is_grasping0):
        group_name = "both_arms"
    elif in_planning[0]:
        group_name = 'left_arm'
    elif in_planning[1]:
        group_name = 'right_arm'
    else:
        raise NoArmsMoving('No arms are moving!')
    return goals, group_name, q0
