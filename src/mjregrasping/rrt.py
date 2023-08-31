from typing import Optional

import numpy as np
from pymjregrasping_cpp import RRTPlanner

import dynamic_reconfigure
import rospy
from dynamic_reconfigure.client import Client
from mjregrasping.grasp_conversions import grasp_locations_to_xpos
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_is_grasping
from mjregrasping.moveit_planning import make_planning_scene
from mjregrasping.physics import Physics, get_full_q
from mjregrasping.viz import Viz
from moveit_msgs.msg import MotionPlanResponse


class GraspRRT:

    def __init__(self):
        self.rrt = RRTPlanner()
        # For some reason, I can fix the RRT collision issues by setting these params
        client = Client("/ompl")
        client.update_configuration({
            "maximum_waypoint_distance": 0.1,
        })

    def plan(self, phy: Physics, strategy, locs: np.ndarray, viz: Optional[Viz], allowed_planning_time=5.0,
             pos_noise=0.05):
        phy_plan = phy.copy_all()
        goals, group_name, q0 = plan_to_grasp(locs, phy_plan, strategy)

        # Visualize the goals
        # if viz:
        #     for i, v in enumerate(goals.values()):
        #         viz.sphere(f'goal_positions/{i}', v, 0.05, [0, 1, 0, 0.2])

        scene_msg = make_planning_scene(phy_plan)
        res: MotionPlanResponse = self.rrt.plan(scene_msg, group_name, goals, bool(viz), allowed_planning_time,
                                                pos_noise)
        return res, scene_msg

    def display_result(self, viz, res, scene_msg):
        viz.rviz.viz_scene(scene_msg)
        self.rrt.display_result(res)

    def is_state_valid(self, phy: Physics):
        scene_msg = make_planning_scene(phy)
        return self.rrt.is_state_valid(scene_msg)


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
        raise ValueError('No arms are moving!')
    return goals, group_name, q0
