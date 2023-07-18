from typing import List

import numpy as np
from numpy.linalg import norm

from mjregrasping.goal_funcs import get_results_common, get_rope_points, get_keypoint, get_regrasp_costs, \
    get_nongrasping_rope_contact_cost
from mjregrasping.goals import MPPIGoal, result, as_floats, as_float
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets, \
    grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.grasping import get_is_grasping, get_finger_qs, get_grasp_locs
from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.regrasp_fsm import Modes, RegraspFSM
from mjregrasping.viz import Viz


class RegraspGoal(MPPIGoal):

    def __init__(self, op_goal, skeletons, grasp_goal_radius, viz: Viz, phy: Physics):
        super().__init__(viz)
        self.op_goal = op_goal
        self.skeletons = skeletons
        self.grasp_goal_radius = grasp_goal_radius
        self.maintain_locs = None
        current_locs = get_grasp_locs(phy)
        gripper_names = phy.o.rd.rope_grasp_eqs
        self.fsm = RegraspFSM(gripper_names, current_locs, None, viz)

    def satisfied(self, phy: Physics):
        return self.op_goal.satisfied(phy)

    def viz_goal(self, phy: Physics):
        self.op_goal.viz_goal(phy)

    def get_results(self, phy: Physics):
        # Create goals that makes the closest point on the rope the intended goal
        # For create a result tuple for each gripper, based on gripper_action
        # The only case where we want to make a gripper result/cost is when we are not currently grasping
        # but when gripper_action is 1, meaning change the grasp state.
        tools_pos, joint_positions, contact_cost, is_unstable = get_results_common(phy)
        is_grasping = get_is_grasping(phy)
        rope_points = get_rope_points(phy)
        finger_qs = get_finger_qs(phy)
        op_goal_body_idx, op_goal_offset = grasp_locations_to_indices_and_offsets(self.op_goal.loc, phy)
        keypoint = get_keypoint(phy, op_goal_body_idx, op_goal_offset)

        current_locs = get_grasp_locs(phy)

        grasp_locs = self.fsm.locs

        nongrasping_rope_contact_cost = get_nongrasping_rope_contact_cost(phy, grasp_locs)

        _, _, grasp_xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy, grasp_locs)

        return result(tools_pos, contact_cost, is_grasping, current_locs, is_unstable, rope_points, keypoint,
                      finger_qs, grasp_locs, grasp_xpos, joint_positions, nongrasping_rope_contact_cost)

    def cost(self, results):
        (tools_pos, contact_cost, is_grasping, current_locs, is_unstable, rope_points, keypoint, finger_qs,
         grasp_locs, grasp_xpos, joint_positions, nongrasping_rope_contact_cost) = as_floats(results)
        keypoint_dist = norm(keypoint - self.op_goal.goal_point, axis=-1)

        unstable_cost = is_unstable * hp['unstable_weight']

        goal_cost = keypoint_dist * hp['goal_weight']

        # NOTE: reading class variables from multiple processes without any protection!
        grasp_finger_cost, grasp_pos_cost, grasp_near_cost = get_regrasp_costs(finger_qs, is_grasping,
                                                                               current_locs, grasp_locs,
                                                                               grasp_xpos, tools_pos, rope_points)

        # penalize distance of q's from 0
        # TODO: only do this for joints which are not participating in any grasps
        home_cost = np.sum(np.abs(joint_positions)) * hp['home_weight']

        if self.fsm.mode == Modes.REGRASPING:
            w_goal = 0
        else:
            w_goal = 1
        return (
            contact_cost,
            unstable_cost,
            w_goal * goal_cost,
            # split up grasp costs by gripper?
            grasp_finger_cost,
            grasp_pos_cost,
            grasp_near_cost,
            home_cost,
            nongrasping_rope_contact_cost,
        )

    @staticmethod
    def cost_names():
        return [
            "contact",
            "unstable",
            "goal",
            "grasp_finger",
            "grasp_pos",
            "grasp_near",
            "home",
            "nongrasping_rope_contact",
        ]

    def viz_result(self, phy: Physics, result, idx: int, scale, color):
        tools_pos = as_float(result[0])
        keypoints = as_float(result[6])
        t0 = 0
        self.viz_ee_lines(tools_pos, idx, scale, color)
        self.viz_rope_lines(keypoints, idx, scale, color='y')

        grasp_xpos = as_float(result[9])[t0]
        for name, xpos in zip(phy.o.rd.rope_grasp_eqs, grasp_xpos):
            self.viz.sphere(f'{name}_xpos', xpos, radius=hp['grasp_goal_radius'], color=(0, 1, 0, 0.4),
                            frame_id='world', idx=0)

    def update_fsms(self, phy: Physics, is_stuck: List[bool], planner: HomotopyRegraspPlanner):
        if self.maintain_locs is None:
            self.maintain_locs = get_grasp_locs(phy)

        current_locs = get_grasp_locs(phy)

        needs_warmstart = self.fsm.update_state(phy, is_stuck, current_locs, planner)
        return needs_warmstart

