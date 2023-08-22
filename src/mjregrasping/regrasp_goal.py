import numpy as np
from numpy.linalg import norm

from mjregrasping.goal_funcs import get_results_common, get_rope_points, get_keypoint, get_regrasp_costs, \
    get_nongrasping_rope_contact_cost
from mjregrasping.goals import MPPIGoal, result, as_floats, as_float, GraspLocsGoal, ObjectPointGoalBase
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets, \
    grasp_locations_to_xpos
from mjregrasping.grasping import get_is_grasping, get_finger_qs, get_grasp_locs
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.viz import Viz


class RegraspGoal(MPPIGoal):

    def __init__(self, op_goal: ObjectPointGoalBase, grasp_goal: GraspLocsGoal, grasp_goal_radius, viz: Viz):
        super().__init__(viz)
        self.op_goal = op_goal
        self.grasp_goal_radius = grasp_goal_radius
        self.grasp_goal = grasp_goal

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

        grasp_locs = self.grasp_goal.get_grasp_locs()

        nongrasping_rope_contact_cost = get_nongrasping_rope_contact_cost(phy, grasp_locs)

        grasp_xpos = grasp_locations_to_xpos(phy, grasp_locs)

        return result(tools_pos, contact_cost, is_grasping, current_locs, is_unstable, rope_points, keypoint,
                      finger_qs, grasp_locs, grasp_xpos, joint_positions, nongrasping_rope_contact_cost)

    def costs(self, results, u_sample):
        (tools_pos, contact_cost, is_grasping, current_locs, is_unstable, rope_points, keypoint, finger_qs,
         grasp_locs, grasp_xpos, joint_positions, nongrasping_rope_contact_cost) = as_floats(results)
        op_goal_cost = self.op_goal.cost(rope_points, keypoint)

        unstable_cost = is_unstable * hp['unstable_weight']

        goal_cost = op_goal_cost * hp['goal_weight']

        nongrasping_rope_contact_cost = nongrasping_rope_contact_cost * hp['nongrasping_rope_contact_weight']

        # NOTE: reading class variables from multiple processes without any protection!
        grasp_finger_cost, grasp_pos_cost, grasp_near_cost = get_regrasp_costs(finger_qs, is_grasping,
                                                                               current_locs, grasp_locs,
                                                                               grasp_xpos, tools_pos, rope_points)

        gripper_to_goal_cost = np.sum(norm(tools_pos - self.op_goal.goal_point, axis=-1) * is_grasping, axis=-1)
        gripper_to_goal_cost = gripper_to_goal_cost * hp['gripper_to_goal_weight']

        desired_q = np.zeros_like(joint_positions)
        desired_q[1] = 0.5
        desired_q_cost = np.sum(np.abs(joint_positions - desired_q), axis=-1) * hp['home_weight']

        # FIXME: do a better smoothness cost that is more than just one time step, do the full correlation or something
        #  also how do we treat magnitude vs direction?
        u_diff_normalized = (u_sample[1:] - u_sample[:-1])
        smoothness_costs = norm(u_diff_normalized, axis=-1)
        smoothness_cost = smoothness_costs * hp['smoothness_weight']

        contact_cost = sum(contact_cost)
        unstable_cost = sum(unstable_cost)
        goal_cost = sum(goal_cost)
        grasp_finger_cost = sum(grasp_finger_cost)
        grasp_pos_cost = sum(grasp_pos_cost)
        grasp_near_cost = sum(grasp_near_cost)
        desired_q_cost = sum(desired_q_cost)
        nongrasping_rope_contact_cost = sum(nongrasping_rope_contact_cost)
        gripper_to_goal_cost = sum(gripper_to_goal_cost)
        smoothness_cost = sum(smoothness_cost)

        is_grasping = as_float(results[2])
        no_gripper_grasping = np.any(np.all(np.logical_not(is_grasping), axis=-1), axis=-1)
        ever_not_grasping_cost = no_gripper_grasping * hp['ever_not_grasping_weight']

        return (
            contact_cost,
            unstable_cost,
            goal_cost,
            # split up grasp costs by gripper?
            grasp_finger_cost,
            grasp_pos_cost,
            grasp_near_cost,
            desired_q_cost,
            nongrasping_rope_contact_cost,
            gripper_to_goal_cost,
            ever_not_grasping_cost,
            smoothness_cost,
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
            "desired_q",
            "nongrasping_rope_contact",
            "gripper_to_goal",
            "ever_not_grasping",
            "smoothness",
        ]

    def viz_result(self, phy: Physics, result, idx: int, scale, color):
        tools_pos = as_float(result[0])
        keypoints = as_float(result[6])
        t0 = 0
        self.viz_ee_lines(tools_pos, idx, scale, color)
        self.viz_rope_lines(keypoints, idx, scale, color='y')

        # grasp_xpos = as_float(result[9])[t0]
        # for name, xpos in zip(phy.o.rd.rope_grasp_eqs, grasp_xpos):
        #     self.viz.sphere(f'{name}_xpos', xpos, radius=hp['grasp_goal_radius'], color=(0, 1, 0, 0.4), idx=0,
        #                     frame_id='world')
