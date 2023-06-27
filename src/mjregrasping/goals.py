import logging
from copy import deepcopy

import mujoco
import numpy as np
import rerun as rr
from numpy.linalg import norm

from mjregrasping.body_with_children import Objects
from mjregrasping.goal_funcs import get_action_cost, get_results_common, get_rope_points, \
    get_keypoint
from mjregrasping.grasp_state_utils import grasp_locations_to_indices_and_offsets, \
    grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.grasping import get_is_grasping, get_finger_qs
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.viz import Viz

logger = logging.getLogger(f'rosout.{__name__}')


def as_floats(results):
    """ results is a numpy array of shape K, where each element can be viewed as a [B, T, ...] matrix """
    return [as_float(result_i) for result_i in results]


def as_float(result_i):
    # FIXME: I'm not sure why tolist() is needed here
    if isinstance(result_i, np.ndarray):
        return np.array(result_i.tolist(), dtype=float)
    else:
        return np.array(result_i, dtype=float)


def result(*results):
    """ we need to return a copy to detach from the simulator state which is mutated in-place """
    return deepcopy(np.array(results, dtype=object))


def softmax(x, temp):
    x = x / temp
    return np.exp(x) / np.exp(x).sum(-1, keepdims=True)


class MPPIGoal:

    def __init__(self, viz: Viz):
        self.viz = viz

    def cost(self, results):
        """
        Args:
            results: the output of get_results()

        Returns:
            matrix of costs [b, horizon]

        """
        # TODO: tag costs as either primary or secondary, and soft or hard
        # so we can use them to detect being "stuck"
        raise NotImplementedError()

    def costs(self, results):
        # TODO: tag costs as either primary or secondary, and soft or hard
        #  override this method not "cost()"
        raise NotImplementedError()

    def satisfied(self, phy):
        raise NotImplementedError()

    def viz_sphere(self, position, radius):
        self.viz.sphere(ns='goal', position=position, radius=radius, frame_id='world', color=[1, 0, 1, 0.5], idx=0)
        self.viz.tf(translation=position, quat_xyzw=[0, 0, 0, 1], parent='world', child='goal')

    def viz_result(self, result, idx: int, scale, color):
        raise NotImplementedError()

    def viz_ee_lines(self, left_tool_pos, right_tool_pos, idx: int, scale: float, color):
        self.viz.lines(left_tool_pos, ns='left_ee', idx=idx, scale=scale, color=color)
        self.viz.lines(right_tool_pos, ns='right_ee', idx=idx, scale=scale, color=color)

    def viz_rope_lines(self, rope_pos, idx: int, scale: float, color):
        self.viz.lines(rope_pos, ns='rope', idx=idx, scale=scale, color=color)

    def get_results(self, phy):
        """
        Returns: the result is any object or tuple of objects, and will be passed to cost()
        """
        raise NotImplementedError()

    def viz_goal(self, phy):
        raise NotImplementedError()


class GraspRopeGoal(MPPIGoal):

    def __init__(self, body_id_to_grasp: int, offset: float, goal_radius: float, gripper_idx: int,
                 objects: Objects,
                 viz: Viz):
        super().__init__(viz)
        self.body_id_to_grasp = body_id_to_grasp
        self.offset = offset
        self.goal_radius = goal_radius
        self.gripper_idx = gripper_idx
        self.objects = objects
        self.initial_body_pos = None

    def get_results(self, phy):
        left_tool_pos, right_tool_pos, joint_positions, contact_cost, is_unstable = get_results_common(self.objects,
                                                                                                       phy)
        body_pos = self.get_body_pos(phy.d)
        return result(left_tool_pos, right_tool_pos, joint_positions, body_pos, contact_cost, is_unstable)

    def cost(self, results):
        left_gripper_pos, right_gripper_pos, joint_positions, body_pos, contact_cost, is_unstable = as_floats(results)
        pred_contact_cost = contact_cost[:, 1:]
        gripper_point = self.choose_gripper_pos(left_gripper_pos, right_gripper_pos)
        dist_cost = norm(body_pos - gripper_point, axis=-1)[:, 1:]
        action_cost = get_action_cost(joint_positions)

        if self.initial_body_pos is None:
            self.initial_body_pos = body_pos
        rope_motion_cost = norm(body_pos - self.initial_body_pos, axis=-1)[:, 1:] * hp['rope_motion_weight']

        unstable_cost = is_unstable[:, 1:] * hp['unstable_weight']

        cost = pred_contact_cost + dist_cost + action_cost + rope_motion_cost + unstable_cost

        rr.log_scalar('grasp_rope_goal/pred_contact', pred_contact_cost.mean(), color=[255, 255, 0])
        rr.log_scalar('grasp_rope_goal/rope_motion', rope_motion_cost.mean(), color=[255, 0, 0])
        rr.log_scalar('grasp_rope_goal/action', action_cost.mean(), color=[255, 255, 255])
        rr.log_scalar('grasp_rope_goal/dist', dist_cost.mean(), color=[0, 255, 0])

        return cost

    def satisfied(self, phy):
        body_pos = self.get_body_pos(phy.d)
        gripper_pos = self.choose_gripper_pos(phy.d.site('left_tool').xpos, phy.d.site('right_tool').xpos)
        distance = norm(body_pos - gripper_pos, axis=-1)
        return distance < self.goal_radius

    def viz_goal(self, phy):
        body_pos = self.get_body_pos(phy.d)
        self.viz_sphere(body_pos, self.goal_radius)

    def viz_result(self, result, idx: int, scale, color):
        left_tool_pos = as_float(result[0])
        right_tool_pos = as_float(result[1])
        self.viz_ee_lines(left_tool_pos, right_tool_pos, idx, scale, color)
        rope_pos = as_float(result[3])
        self.viz_rope_lines(rope_pos, idx, scale, color='y')

    def get_body_pos(self, data):
        body_pos = data.xpos[self.body_id_to_grasp]
        # add offset
        body_x_axis_in_world_frame = data.xmat[self.body_id_to_grasp].reshape(3, 3)[:, 0]
        return body_pos + self.offset * body_x_axis_in_world_frame

    def choose_gripper_pos(self, left_gripper_pos, right_gripper_pos):
        if self.gripper_idx == 0:
            gripper_point = left_gripper_pos
        elif self.gripper_idx == 1:
            gripper_point = right_gripper_pos
        else:
            raise ValueError(f"unknown gripper_idx {self.gripper_idx}")
        return gripper_point


class ObjectPointGoal(MPPIGoal):

    def __init__(self, goal_point: np.array, goal_radius: float, loc: float, objects, viz: Viz):
        super().__init__(viz)
        self.goal_point = goal_point
        self.loc = loc
        self.goal_radius = goal_radius
        self.objects = objects
        self.rope_body_indices = objects.rope.body_indices
        self.body_idx, self.offset = grasp_locations_to_indices_and_offsets(loc, self.rope_body_indices)

    def get_results(self, phy):
        left_tool_pos, right_tool_pos, joint_positions, contact_cost, is_unstable = get_results_common(self.objects,
                                                                                                       phy)
        rope_points = get_rope_points(phy, self.objects.rope.body_indices)
        eq_indices = [
            mujoco.mj_name2id(phy.m, mujoco.mjtObj.mjOBJ_EQUALITY, 'left'),
            mujoco.mj_name2id(phy.m, mujoco.mjtObj.mjOBJ_EQUALITY, 'right'),
        ]
        is_grasping = phy.m.eq_active[eq_indices]

        keypoint = get_keypoint(phy, self.body_idx, self.offset)

        return result(rope_points, keypoint, joint_positions, left_tool_pos, right_tool_pos, is_grasping, contact_cost,
                      is_unstable)

    def cost(self, results):
        rope_points, keypoint, joint_positions, left_tool_pos, right_tool_pos, is_grasping, contact_cost, is_unstable = as_floats(
            results)

        pred_contact_cost = contact_cost[:, 1:]
        pred_rope_points = rope_points[:, 1:]
        keypoint_dist = self.keypoint_dist_to_goal(keypoint)
        pred_keypoint_dist = keypoint_dist[:, 1:]
        gripper_points = np.stack([left_tool_pos, right_tool_pos], axis=-2)
        pred_gripper_points = gripper_points[:, 1:]
        pred_joint_positions = joint_positions[:, 1:]

        pred_is_grasping = is_grasping[:, 1:]  # skip t=0

        unstable_cost = is_unstable[:, 1:] * hp['unstable_weight']

        point_dist_cost = pred_keypoint_dist * hp['point_dist_weight']

        # Add cost for grippers that are not grasping
        # that encourages them to remain close to the rope
        # [b, horizon, n_rope_points, n_grippers]
        rope_gripper_dists = norm(pred_gripper_points[..., None, :, :] - pred_rope_points[..., None, :], axis=-1)
        pred_is_not_grasping = 1 - pred_is_grasping
        min_nongrasping_dists = np.sum(np.min(rope_gripper_dists, -2) * pred_is_not_grasping, -1)  # [b, horizon]
        min_nongrasping_dists = np.sqrt(np.maximum(min_nongrasping_dists - hp['nongrasping_close'], 0))

        min_nongrasping_cost = min_nongrasping_dists * hp['min_nongrasping_rope_gripper_dists']

        # Add a cost that non-grasping grippers should try to return to a "home" position.
        # Home is assumed to be 0, so penalize the distance from 0.
        # FIXME: doesn't generalize, hard-coded for Val
        arm_gripper_matrix = np.zeros([18, 2])
        left_joint_indices = np.arange(2, 10)
        right_joint_indices = np.arange(10, 17)
        arm_gripper_matrix[left_joint_indices, 0] = 1
        arm_gripper_matrix[right_joint_indices, 1] = 1
        home_cost_joints = np.abs(pred_joint_positions)  # [b, horizon, n_joints]
        home_cost_grippers = home_cost_joints @ arm_gripper_matrix
        nongrasping_home_cost = np.sum(home_cost_grippers * pred_is_not_grasping, -1)  # [b, horizon]
        nongrasping_home_cost = nongrasping_home_cost * hp['nongrasping_home']

        action_cost = get_action_cost(pred_joint_positions)

        cost = point_dist_cost + pred_contact_cost + unstable_cost + nongrasping_home_cost + action_cost + min_nongrasping_cost

        # keep track of this in a member variable, so we can detect when it's value has changed
        rr.log_scalar('object_point_goal/points', point_dist_cost.mean(), color=[0, 0, 255])
        rr.log_scalar('object_point_goal/pred_contact', pred_contact_cost.mean(), color=[255, 255, 0])
        rr.log_scalar('object_point_goal/min_nongrasping', min_nongrasping_cost.mean(), color=[0, 255, 255])
        rr.log_scalar('object_point_goal/action', action_cost.mean(), color=[255, 255, 255])
        rr.log_scalar('object_point_goal/home', nongrasping_home_cost.mean(), color=[128, 0, 0])

        return cost  # [b, horizon]

    def satisfied(self, phy):
        keypoint = get_keypoint(phy, self.body_idx, self.offset)
        error = self.keypoint_dist_to_goal(keypoint).squeeze()
        return error < self.goal_radius

    def gripper_dists_to_goal(self, left_tool_pos, right_tool_pos):
        gripper_points = np.stack([left_tool_pos, right_tool_pos], axis=-2)
        gripper_distances = norm((gripper_points - self.goal_point[None, None, None]), axis=-1)
        return gripper_distances

    def keypoint_dist_to_goal(self, keypoint):
        return norm((keypoint - self.goal_point), axis=-1)

    def viz_result(self, result, idx: int, scale, color):
        left_tool_pos = as_float(result[2])
        right_tool_pos = as_float(result[3])
        keypoints = as_float(result[0])
        self.viz_ee_lines(left_tool_pos, right_tool_pos, idx, scale, color)
        self.viz_rope_lines(keypoints, idx, scale, color='y')

    def viz_goal(self, phy):
        self.viz_sphere(self.goal_point, self.goal_radius)


def get_finger_cost(finger_qs, desired_is_grasping):
    desired_finger_qs = np.array(
        [hp['finger_q_closed'] if is_g_i else hp['finger_q_open'] for is_g_i in desired_is_grasping])
    finger_cost = (np.sum(np.abs(finger_qs - desired_finger_qs), axis=-1))
    return finger_cost


class RegraspGoal(MPPIGoal):

    def __init__(self, op_goal, grasp_goal_radius, objects, viz: Viz):
        super().__init__(viz)
        self.op_goal = op_goal
        self.objects = objects
        self.grasp_goal_radius = grasp_goal_radius
        self.n_g = hp['n_g']

    def satisfied(self, phy):
        return self.op_goal.satisfied(phy)

    def viz_goal(self, phy):
        self.op_goal.viz_goal(phy)

    def get_results(self, phy: Physics):
        # Create goals that makes the closest point on the rope the intended goal
        # For create a result tuple for each gripper, based on gripper_action
        # The only case where we want to make a gripper result/cost is when we are not currently grasping
        # but when gripper_action is 1, meaning change the grasp state.
        left_tool_pos, right_tool_pos, _, contact_cost, is_unstable = get_results_common(self.objects, phy)
        is_grasping = get_is_grasping(phy.m)
        rope_points = get_rope_points(phy, self.objects.rope.body_indices)
        finger_qs = get_finger_qs(phy)
        keypoint = get_keypoint(phy, self.op_goal.body_idx, self.op_goal.offset)

        desired_regrasp_locs = np.array([self.viz.p.left_regrasp_point, self.viz.p.right_regrasp_point])
        _, _, desired_regrasp_xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy, desired_regrasp_locs,
                                                                                     self.objects.rope.body_indices)

        return result(left_tool_pos, right_tool_pos, contact_cost, is_grasping, is_unstable, rope_points, keypoint,
                      finger_qs, desired_regrasp_xpos)

    def cost(self, results, is_grasping0):
        w_goal = self.viz.p.config['w_goal']
        w_regrasp = self.viz.p.config['w_regrasp_point']
        desired_grasp_locs = np.array([self.viz.p.left_regrasp_point, self.viz.p.right_regrasp_point])

        # TODO: return matrix of tool positions, this assumes 2 grippers
        left_tool_pos, right_tool_pos, contact_cost, is_grasping, is_unstable, rope_points, keypoint, finger_qs, desired_regrasp_xpos = as_floats(
            results)

        desired_is_grasping = desired_grasp_locs > 0

        keypoint_dist = norm(keypoint - self.op_goal.goal_point, axis=-1)

        unstable_cost = is_unstable * hp['unstable_weight']

        goal_cost = keypoint_dist * hp['goal_weight']

        maintain_grasps_cost = get_finger_cost(finger_qs, is_grasping0) * hp['finger_weight']

        # Exploration costs
        regrasp_finger_cost = get_finger_cost(finger_qs, desired_is_grasping) * hp['finger_weight']

        tool_pos = np.stack([left_tool_pos, right_tool_pos], axis=0)  # [n_g, 3]
        regrasp_dists = norm(desired_regrasp_xpos - tool_pos, axis=-1)
        needs_grasp = desired_is_grasping * (1 - is_grasping)
        regrasp_pos_cost = np.sum(regrasp_dists * needs_grasp, -1) * hp['regrasp_weight']

        return (
            contact_cost,
            unstable_cost,
            w_goal * goal_cost,
            w_goal * maintain_grasps_cost,
            w_regrasp * regrasp_finger_cost,
            w_regrasp * regrasp_pos_cost,
        )

    def cost_names(self):
        return [
            "contact",
            "unstable",
            "goal",
            "maintain_grasps_cost",
            "regrasp_finger_cost",
            "regrasp_pos_cost",
        ]

    def viz_result(self, result, idx: int, scale, color):
        left_tool_pos = as_float(result[0])
        right_tool_pos = as_float(result[1])
        keypoints = as_float(result[6])
        self.viz_ee_lines(left_tool_pos, right_tool_pos, idx, scale, color)
        self.viz_rope_lines(keypoints, idx, scale, color='y')
