import logging
from copy import copy

import mujoco
import numpy as np
import rerun as rr
from matplotlib import cm
from numpy.linalg import norm

from mjregrasping.body_with_children import Objects
from mjregrasping.grasp_state import grasp_indices_to_locations
from mjregrasping.my_transforms import angle_between
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.ring_utils import make_ring_mat
from mjregrasping.rviz import plot_spheres_rviz
from mjregrasping.viz import Viz

logger = logging.getLogger(f'rosout.{__name__}')


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


class GripperPointGoal(MPPIGoal):

    def __init__(self, goal_point: np.array, goal_radius: float, gripper_idx: int, objects: Objects,
                 viz: Viz):
        super().__init__(viz)
        self.goal_point = goal_point
        self.goal_radius = goal_radius
        self.gripper_idx = gripper_idx
        self.objects = objects

    def cost(self, results):
        """
        Args:
            results: the output of get_results()

        Returns:
            matrix of costs [b, horizon]

        """
        left_gripper_pos, right_gripper_pos, joint_positions, contact_cost = results
        pred_contact_cost = contact_cost[:, 1:]
        gripper_point = self.choose_gripper_pos(left_gripper_pos, right_gripper_pos)
        dist_cost = norm(self.goal_point - gripper_point, axis=-1)[:, 1:]
        action_cost = get_action_cost(joint_positions)

        cost = copy(pred_contact_cost)
        cost += dist_cost
        cost += action_cost

        rr.log_scalar('gripper_point_goal/pred_contact', pred_contact_cost.mean(), color=[255, 255, 0])
        rr.log_scalar('gripper_point_goal/action', action_cost.mean(), color=[255, 255, 255])
        rr.log_scalar('gripper_point_goal/dist', dist_cost.mean())

        return cost

    def satisfied(self, phy):
        gripper_pos = self.choose_gripper_pos(phy.d.site('left_tool').xpos, phy.d.site('right_tool').xpos)
        distance = norm(self.goal_point - gripper_pos, axis=-1)
        return distance < self.goal_radius

    def viz_goal(self, phy):
        self.viz_sphere(self.goal_point, self.goal_radius)

    def get_results(self, phy):
        """
        Returns: the result is any object or tuple of objects, and will be passed to cost()

        """
        joint_indices_for_actuators = phy.m.actuator_trnid[:, 0]
        joint_positions = phy.d.qpos[joint_indices_for_actuators]
        contact_cost = get_contact_cost(phy, self.objects)
        return phy.d.site('left_tool').xpos, phy.d.site('right_tool').xpos, joint_positions, contact_cost

    def viz_result(self, result, idx: int, scale, color):
        left_tool_pos = result[0]
        right_tool_pos = result[1]
        self.viz_ee_lines(left_tool_pos, right_tool_pos, idx, scale, color)

    def choose_gripper_pos(self, left_gripper_pos, right_gripper_pos):
        if self.gripper_idx == 0:
            gripper_point = left_gripper_pos
        elif self.gripper_idx == 1:
            gripper_point = right_gripper_pos
        else:
            raise ValueError(f"unknown gripper_idx {self.gripper_idx}")
        return gripper_point


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
        left_tool_pos, right_tool_pos, joint_positions, contact_cost = get_results_common(self.objects, phy)
        body_pos = self.get_body_pos(phy.d)
        return left_tool_pos, right_tool_pos, joint_positions, body_pos, contact_cost

    def cost(self, results):
        left_gripper_pos, right_gripper_pos, joint_positions, body_pos, contact_cost = results
        pred_contact_cost = contact_cost[:, 1:]
        gripper_point = self.choose_gripper_pos(left_gripper_pos, right_gripper_pos)
        dist_cost = norm(body_pos - gripper_point, axis=-1)[:, 1:]
        action_cost = get_action_cost(joint_positions)

        if self.initial_body_pos is None:
            self.initial_body_pos = body_pos
        rope_motion_cost = norm(body_pos - self.initial_body_pos, axis=-1)[:, 1:] * hp['rope_motion_weight']

        cost = copy(pred_contact_cost)
        cost += dist_cost
        cost += action_cost
        cost += rope_motion_cost

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
        left_tool_pos = result[0]
        right_tool_pos = result[1]
        self.viz_ee_lines(left_tool_pos, right_tool_pos, idx, scale, color)
        rope_pos = np.array(result[3])
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

    def __init__(self, dfield, goal_point: np.array, goal_radius: float, body_idx: int, objects, viz: Viz):
        super().__init__(viz)
        self.goal_point = goal_point
        self.dfield = dfield
        self.body_idx = body_idx
        self.goal_radius = goal_radius
        self.objects = objects

    def get_results(self, phy):
        left_tool_pos, right_tool_pos, joint_positions, contact_cost = get_results_common(self.objects, phy)
        rope_points = get_rope_points(phy, self.objects.rope.body_indices)
        eq_indices = [
            mujoco.mj_name2id(phy.m, mujoco.mjtObj.mjOBJ_EQUALITY, 'left'),
            mujoco.mj_name2id(phy.m, mujoco.mjtObj.mjOBJ_EQUALITY, 'right'),
        ]
        is_grasping = phy.m.eq_active[eq_indices]

        return rope_points, joint_positions, left_tool_pos, right_tool_pos, is_grasping, contact_cost

    def point_dist_cost(self, results):
        rope_points = results[0]
        point_dist = self.min_dist_to_specified_point(rope_points)
        pred_point_dist = point_dist[:, 1:]

        return pred_point_dist

    def cost(self, results):
        rope_points, joint_positions, left_tool_pos, right_tool_pos, is_grasping, contact_cost = results

        pred_rope_points = rope_points[:, 1:]
        pred_contact_cost = contact_cost[:, 1:]
        point_dist = self.min_dist_to_specified_point(rope_points)
        pred_point_dist = point_dist[:, 1:]
        gripper_points = np.stack([left_tool_pos, right_tool_pos], axis=-2)
        pred_gripper_points = gripper_points[:, 1:]
        pred_joint_positions = joint_positions[:, 1:]

        pred_is_grasping = is_grasping[:, 1:]  # skip t=0

        # Get the potential field gradient at the specified points
        gripper_dfield = self.dfield.get_costs(pred_gripper_points)  # [b, h, 2]
        gripper_dfield *= hp['gripper_dfield']
        grasping_gripper_dfield = np.sum(gripper_dfield * pred_is_grasping, -1)  # [b, horizon]

        cost = copy(pred_point_dist) * hp['point_dist_weight']

        cost += grasping_gripper_dfield

        cost += pred_contact_cost

        # Add cost for grippers that are not grasping
        # that encourages them to remain close to the rope
        # [b, horizon, n_rope_points, n_grippers]
        rope_gripper_dists = norm(pred_gripper_points[..., None, :, :] - pred_rope_points[..., None, :], axis=-1)
        pred_is_not_grasping = 1 - pred_is_grasping
        min_nongrasping_dists = np.sum(np.min(rope_gripper_dists, -2) * pred_is_not_grasping, -1)  # [b, horizon]
        min_nongrasping_dists = np.sqrt(np.maximum(min_nongrasping_dists - hp['nongrasping_close'], 0))

        min_nongrasping_cost = min_nongrasping_dists * hp['min_nongrasping_rope_gripper_dists']
        cost += min_nongrasping_cost

        # Add a cost that non-grasping grippers should try to return to a "home" position.
        # Home is assumed to be 0, so penalize the distance from 0.
        # FIXME: doesn't generalize, hard-coded for Val
        arm_gripper_matrix = np.zeros([18, 2])
        left_joint_indices = np.arange(2, 10)
        right_joint_indices = np.arange(10, 19)
        arm_gripper_matrix[left_joint_indices, 0] = 1
        arm_gripper_matrix[right_joint_indices, 1] = 1
        home_cost_joints = np.abs(pred_joint_positions)  # [b, horizon, n_joints]
        home_cost_grippers = home_cost_joints @ arm_gripper_matrix
        nongrasping_home_cost = np.sum(home_cost_grippers * pred_is_not_grasping, -1)  # [b, horizon]
        nongrasping_home_cost = nongrasping_home_cost * hp['nongrasping_home']
        cost += nongrasping_home_cost

        # Add an action cost
        action_cost = get_action_cost(joint_positions)
        cost += action_cost

        # keep track of this in a member variable, so we can detect when it's value has changed
        rr.log_scalar('object_point_goal/points', pred_point_dist.mean(), color=[0, 0, 255])
        rr.log_scalar('object_point_goal/grasping_gripper_dfield', grasping_gripper_dfield.mean(), color=[255, 0, 255])
        rr.log_scalar('object_point_goal/pred_contact', pred_contact_cost.mean(), color=[255, 255, 0])
        rr.log_scalar('object_point_goal/min_nongrasping', min_nongrasping_cost.mean(), color=[0, 255, 255])
        rr.log_scalar('object_point_goal/action', action_cost.mean(), color=[255, 255, 255])
        rr.log_scalar('object_point_goal/home', nongrasping_home_cost.mean(), color=[128, 0, 0])

        return cost  # [b, horizon]

    def satisfied(self, phy):
        rope_points = get_rope_points(phy, self.objects.rope.body_indices)
        error = self.min_dist_to_specified_point(rope_points).squeeze()
        return error < self.goal_radius

    def gripper_dists_to_specified_point(self, left_tool_pos, right_tool_pos):
        gripper_points = np.stack([left_tool_pos, right_tool_pos], axis=-2)
        gripper_distances = norm((gripper_points - self.goal_point[None, None, None]), axis=-1)
        return gripper_distances

    def min_dist_to_specified_point(self, points):
        return self.min_dist_from_points_to_specified_point(points[..., self.body_idx, :])

    def min_dist_from_points_to_specified_point(self, points):
        return norm((points - self.goal_point), axis=-1)

    def viz_result(self, result, idx: int, scale, color):
        left_tool_pos = result[2]
        right_tool_pos = result[3]
        rope_pos = np.array(result[0])[:, self.body_idx]
        self.viz_ee_lines(left_tool_pos, right_tool_pos, idx, scale, color)
        self.viz_rope_lines(rope_pos, idx, scale, color='y')

    def viz_goal(self, phy):
        self.viz_sphere(self.goal_point, self.goal_radius)


class CombinedGoal(ObjectPointGoal):

    def __init__(self, goal_point: np.array, goal_radius: float, body_idx: int, objects, viz: Viz):
        # TODO: get rid of dfield if we stop using it
        super().__init__(None, goal_point, goal_radius, body_idx, objects, viz)
        self.rope_body_indices = np.array(self.objects.rope.body_indices)
        self.n_b = len(self.rope_body_indices)
        self.n_g = hp['n_g']
        # cached intermediate result used by the MPC algorithm in regrasp_mpc.py
        self.grasp_costs = None

    def get_results(self, phy):
        left_tool_pos, right_tool_pos, joint_positions, contact_cost = get_results_common(self.objects, phy)
        rope_points = np.array([phy.d.xpos[rope_body_idx] for rope_body_idx in self.objects.rope.body_indices])
        eq_active = np.concatenate([phy.m.eq("left").active, phy.m.eq("right").active])
        eq_obj2id = np.concatenate([phy.m.eq("left").obj2id, phy.m.eq("right").obj2id])

        return left_tool_pos, right_tool_pos, joint_positions, contact_cost, rope_points, eq_active, eq_obj2id

    def cost(self, results):
        left_pos, right_pos, joint_positions, contact_cost, rope_points, eq_active, eq_obj2id = results
        pred_contact_cost = contact_cost[:, 1:]  # skip t=0

        point_dist = self.min_dist_to_specified_point(rope_points)
        pred_point_dist = point_dist[:, 1:] * hp['point_dist_weight']

        # get the weight average direction from gripper to rope points,
        # where the weight is the normalized geodesic distance from the gripper.
        loc = grasp_indices_to_locations(self.rope_body_indices, eq_obj2id)
        # FIXME: assumes rope length is 1m???
        num_samples, horizon = loc.shape[:2]
        linespaced = np.tile(np.linspace(0, 1, self.n_b)[None, None, None], [num_samples, horizon, self.n_g, 1])
        inv_geodesic = np.square(1 - (linespaced - loc[..., None]))  # [num_samples, horizon, n_g, n_b]
        weight = softmax(inv_geodesic, 0.5)
        rope_deltas = rope_points[:, :, 1:] - rope_points[:, :, :-1]  # [num_samples, horizon, n_b-1, 3]
        # extend the last delta to be the same as the second last delta
        rope_deltas = np.insert(rope_deltas, -1, rope_deltas[:, :, -1], axis=2)  # [num_samples, horizon, n_b, 3]
        # tile to match the number of grippers
        rope_deltas = np.tile(rope_deltas[:, :, None], [1, 1, self.n_g, 1, 1])  # [num_samples, horizon, n_g, n_b, 3]
        # negate the delta if the point grasped by the gripper is after before the point on the rope
        eq_loc = grasp_indices_to_locations(self.rope_body_indices, eq_obj2id)
        flip = (linespaced > eq_loc[..., None])[..., None]
        rope_deltas = np.where(flip, -rope_deltas, rope_deltas)
        # we need rope deltas to have shape [num_samples, horizon, n_b, 3, 2]
        rope_deltas_weighted = np.einsum('bhgr,bhgrx->bhgx', weight, rope_deltas)  # [num_samples, horizon, n_g, 3]

        # compute alignment between the change in the gripper position and the weighted direction of the rope points
        # and mask basked on whether the grasp is active
        gripper_pos = np.stack([left_pos, right_pos], axis=-2)  # [b, horizon, n_g, 3]
        gripper_dir = gripper_pos[:, 1:] - gripper_pos[:, :-1]  # [num_samples, horizon - 1, n_g, 3]
        gripper_dir = np.insert(gripper_dir, 0, gripper_dir[:, 0], axis=1)  # [num_samples, horizon, n_g, 3]
        pull_cost = angle_between(rope_deltas_weighted, gripper_dir)  # [num_samples, horizon, n_g]
        grasping_pull_cost = np.einsum('bhg,bhg->bh', eq_active, pull_cost)
        pred_grasping_pull_cost = grasping_pull_cost[:, 1:] * hp['pull_cost_weight']

        action_cost = get_action_cost(joint_positions)

        goal_costs = pred_point_dist + pred_grasping_pull_cost

        grasp_w, weighted_grasp_costs = self.compute_weighted_grasp_costs(eq_active, eq_obj2id, left_pos, right_pos,
                                                                          rope_points)
        total_costs = goal_costs + weighted_grasp_costs + action_cost + pred_contact_cost

        rr.log_scalar('grasp_rope_goal/dist', pred_point_dist.mean(), color=[0, 255, 0])
        rr.log_scalar('object_point_goal/points', pred_point_dist.mean(), color=[0, 0, 255])
        rr.log_scalar('object_point_goal/pull_cost', pred_grasping_pull_cost.mean(), color=[255, 0, 255])
        rr.log_scalar('combined_goal/pred_contact', pred_contact_cost.mean(), color=[255, 255, 0])
        rr.log_scalar('combined_goal/action', action_cost.mean(), color=[255, 255, 255])
        rr.log_scalar('combined_goal/goal', goal_costs.mean(), color=[128, 255, 128])
        rr.log_scalar('combined_goal/grasping', self.grasp_costs.mean(), color=[0, 0, 255])
        rr.log_scalar('combined_goal/weighted_grasping', weighted_grasp_costs.mean(), color=[0, 255, 255])
        rr.log_scalar('combined_goal/total', total_costs.mean(), color=[0, 255, 0])

        self.viz_grasp_weights(grasp_w, rope_points)

        return total_costs  # [b, horizon]

    def compute_weighted_grasp_costs(self, eq_active, eq_obj2id, left_pos, right_pos, rope_points):
        self.grasp_costs = np.zeros([self.n_g, self.n_b, hp['num_samples'], hp['horizon']])
        is_grasping_mat = np.zeros([self.n_g, self.n_b])
        for gripper_i in range(self.n_g):
            gripper_pos = left_pos if gripper_i == 0 else right_pos
            # These two things are constant across samples & time
            eq_i_active = eq_active[0, 0, gripper_i]
            eq_i_obj2id = eq_obj2id[0, 0, gripper_i]
            for body_j, body_idx in enumerate(self.rope_body_indices):
                body_pos = rope_points[:, :, body_j]
                dist_cost = norm(body_pos - gripper_pos, axis=-1)[:, 1:]
                grasp_ij_cost = dist_cost
                self.grasp_costs[gripper_i, body_j] = grasp_ij_cost
                is_grasping_mat[gripper_i, body_j] = eq_i_active and eq_i_obj2id == body_idx
        self.grasp_costs *= hp['grasp_weight']  # overall rescale to make comparable with goal cost
        grasp_w = self.get_grasp_weights()
        grasp_w_not_grasping = grasp_w * (1 - is_grasping_mat)
        weighted_grasp_costs = np.einsum('gr,grbh->bh', grasp_w_not_grasping, self.grasp_costs)
        return grasp_w, weighted_grasp_costs

    def viz_grasp_weights(self, grasp_w, rope_points):
        # Visualize grasp_w
        positions = []
        colors = []
        for gripper_i in range(self.n_g):
            offset = np.array([0.01, 0.01, 0.01]) * gripper_i
            for body_j, body_idx in enumerate(self.rope_body_indices):
                grasp_w_ij = grasp_w[gripper_i, body_j]
                color = cm.RdYlGn(grasp_w_ij)
                color = (color[0], color[1], color[2], 0.4)  # alpha
                pos = rope_points[0, 0, body_j] + offset
                positions.append(pos)
                colors.append(color)
        plot_spheres_rviz(self.viz.markers_pub, positions, colors, 0.015, "world", label="grasp_w")

    def get_grasp_weights(self):
        grasp_weights = np.array(
            [[self.viz.p.config[f'left_w_{i}'] for i in range(self.n_b)],
             [self.viz.p.config[f'right_w_{i}'] for i in range(self.n_b)]]
        )
        return grasp_weights

    def viz_result(self, result, idx: int, scale, color):
        left_pos = result[0]
        right_pos = result[1]
        rope_pos = np.array(result[4])[:, self.body_idx]
        self.viz_ee_lines(left_pos, right_pos, idx, scale, color)
        self.viz_rope_lines(rope_pos, idx, scale, color='y')


class CombinedThreadingGoal(CombinedGoal):
    def __init__(self, goal_point: np.array, goal_dir, goal_radius: float, body_idx: int, objects, viz: Viz):
        super().__init__(goal_point, goal_radius, body_idx, objects, viz)
        self.goal_dir = goal_dir

    def cost(self, results):
        left_pos, right_pos, joint_positions, contact_cost, rope_points, eq_active, eq_obj2id = results
        b, t = left_pos.shape[:2]
        rope_keypoint = rope_points[:, :, self.body_idx]
        rope_keypoint_flat = rope_keypoint.reshape(-1, 3)
        b_dir_flat = compute_threading_dir(self.goal_point, self.goal_dir, self.goal_radius, rope_keypoint_flat)
        b_dir = b_dir_flat.reshape(b, t, 3)
        b_dir = b_dir / norm(b_dir, axis=-1, keepdims=True)

        from mjregrasping.rviz import plot_arrows_rviz
        b_dir_viz = b_dir * 0.05
        plot_arrows_rviz(self.viz.markers_pub, rope_keypoint[0, 0:1], b_dir_viz[0, 0:1], "b_dir", 0, 'b')

        # compare the direction of motion of the rope with b_dir
        # if the rope is pointing in the same direction as b_dir, the cost is 0
        rope_deltas = rope_keypoint[:, 1:] - rope_keypoint[:, :-1]  # [b, t-1, 3]
        rope_deltas = rope_deltas / norm(rope_deltas, axis=-1, keepdims=True)
        thread_dir_costs = angle_between(b_dir[:, :-1], rope_deltas)
        thread_dir_costs = thread_dir_costs * hp['thread_dir_weight']

        # compare the orientation of the rope with b_dir
        if self.body_idx == 0:
            orientation = rope_points[:, :, 1] - rope_points[:, :, 0]
        else:
            orientation = rope_points[:, :, self.body_idx] - rope_points[:, :, self.body_idx - 1]
        orientation_costs = angle_between(orientation, b_dir)
        pred_thread_orient_costs = orientation_costs[:, 1:] * hp['thread_orient_weight']

        pred_contact_cost = contact_cost[:, 1:]  # skip t=0

        point_dist = self.min_dist_to_specified_point(rope_points)
        pred_point_dist = point_dist[:, 1:] * hp['point_dist_weight']

        # get the weight average direction from gripper to rope points,
        # where the weight is the normalized geodesic distance from the gripper.
        loc = grasp_indices_to_locations(self.rope_body_indices, eq_obj2id)
        # FIXME: assumes rope length is 1m???
        num_samples, horizon = loc.shape[:2]
        linespaced = np.tile(np.linspace(0, 1, self.n_b)[None, None, None], [num_samples, horizon, self.n_g, 1])
        inv_geodesic = np.square(1 - (linespaced - loc[..., None]))  # [num_samples, horizon, n_g, n_b]
        weight = softmax(inv_geodesic, 0.5)
        rope_deltas = rope_points[:, :, 1:] - rope_points[:, :, :-1]  # [num_samples, horizon, n_b-1, 3]
        # extend the last delta to be the same as the second last delta
        rope_deltas = np.insert(rope_deltas, -1, rope_deltas[:, :, -1], axis=2)  # [num_samples, horizon, n_b, 3]
        # tile to match the number of grippers
        rope_deltas = np.tile(rope_deltas[:, :, None], [1, 1, self.n_g, 1, 1])  # [num_samples, horizon, n_g, n_b, 3]
        # negate the delta if the point grasped by the gripper is after before the point on the rope
        eq_loc = grasp_indices_to_locations(self.rope_body_indices, eq_obj2id)
        flip = (linespaced > eq_loc[..., None])[..., None]
        rope_deltas = np.where(flip, -rope_deltas, rope_deltas)
        # we need rope deltas to have shape [num_samples, horizon, n_b, 3, 2]
        rope_deltas_weighted = np.einsum('bhgr,bhgrx->bhgx', weight, rope_deltas)  # [num_samples, horizon, n_g, 3]

        action_cost = get_action_cost(joint_positions)

        goal_costs = thread_dir_costs + pred_thread_orient_costs

        grasp_w, weighted_grasp_costs = self.compute_weighted_grasp_costs(eq_active, eq_obj2id, left_pos, right_pos,
                                                                          rope_points)
        total_costs = goal_costs + weighted_grasp_costs + action_cost + pred_contact_cost

        rr.log_scalar('grasp_rope_goal/dist', pred_point_dist.mean(), color=[0, 255, 0])
        rr.log_scalar('threading_goal/thread_dir_cost', thread_dir_costs.mean(), color=[0, 255, 0])
        rr.log_scalar('threading_goal/thread_orient_cost', pred_thread_orient_costs.mean(), color=[0, 0, 255])
        rr.log_scalar('combined_goal/pred_contact', pred_contact_cost.mean(), color=[255, 255, 0])
        rr.log_scalar('combined_goal/action', action_cost.mean(), color=[255, 255, 255])
        rr.log_scalar('combined_goal/goal', goal_costs.mean(), color=[128, 255, 128])
        rr.log_scalar('combined_goal/grasping', self.grasp_costs.mean(), color=[0, 0, 255])
        rr.log_scalar('combined_goal/weighted_grasping', weighted_grasp_costs.mean(), color=[0, 255, 255])
        rr.log_scalar('combined_goal/total', total_costs.mean(), color=[0, 255, 0])

        self.viz_grasp_weights(grasp_w, rope_points)

        return total_costs  # [b, horizon]

    def satisfied(self, phy):
        rope_points = get_rope_points(phy, self.objects.rope.body_indices)
        rope_keypoint = rope_points[self.body_idx]
        return np.linalg.norm(rope_keypoint - self.goal_point) < self.goal_radius

    def viz_goal(self, phy):
        self.viz.ring(self.goal_point, self.goal_dir, self.goal_radius)


def compute_threading_dir(ring_position, ring_z_axis, R, p, I=1, mu=1):
    """
    Compute the direction of a virtual magnetic field that will pull the rope through the ring.

    Args:
        ring_position: [3] The position of the origin of the ring in world coordinates
        ring_z_axis: [3] The axis point out up and out of the ring in world coordinates
        R: [1] radius of the ring in meters.
        p: [b, 3] the positions at which you want to compute the field
        I: [1] current, higher means stronger field. If you only care about direction, using 1 is fine.
        mu: [1] another scalar that controls the strength of the field. If you only care about direction, using 1 is fine.

    Returns:
        [b, 3] the direction of the field at p

    """
    dx, x = compute_ring_vecs(ring_position, ring_z_axis, R)
    # We need a batch cross product
    b = p.shape[0]
    n = dx.shape[0]
    dx_repeated = np.repeat(dx[None], b, axis=0)
    x_repeated = np.repeat(x[None], b, axis=0)
    p_repeated = np.repeat(p[:, None], n, axis=1)
    dp_repeated = p_repeated - x_repeated
    cross_product = np.cross(dx_repeated, dp_repeated)
    db = I * R * cross_product / np.linalg.norm(dp_repeated, axis=-1, keepdims=True) ** 3
    b = np.sum(db, axis=-2)
    b *= mu / (4 * np.pi)

    return b


def compute_ring_vecs(ring_position, ring_z_axis, R):
    """
    Compute the direction sub-quantities needed for compute_threading_dir

    Args:
        ring_position: [3] The position of the origin of the ring in world coordinates
        ring_z_axis: [3] The axis point out up and out of the ring in world coordinates
        R: [1] radius of the ring in meters.

    Returns:
        [n, 3] the direction of the conductor (ring) at each ring point
        [n, 3] the position of the ring points

    """
    delta_angle = 0.1
    angles = np.arange(0, 2 * np.pi, delta_angle)
    zeros = np.zeros_like(angles)
    ones = np.ones_like(angles)
    x = np.stack([R * np.cos(angles), R * np.sin(angles), zeros, ones], -1)
    ring_mat = make_ring_mat(ring_position, ring_z_axis)
    x = (x @ ring_mat.T)[:, :3]
    zeros = np.zeros_like(angles)
    dx = np.stack([-np.sin(angles), np.cos(angles), zeros], -1)
    # only rotate here
    ring_rot = ring_mat.copy()[:3, :3]
    dx = (dx @ ring_rot.T)
    dx = dx / np.linalg.norm(dx, axis=-1, keepdims=True)  # normalize
    return dx, x


def get_contact_cost(phy: Physics, objects: Objects):
    # TODO: use SDF to compute near-contact cost to avoid getting too close
    # doing the contact cost calculation here means we don't need to return the entire data.contact array,
    # which makes things simpler and possibly faster, since this operation can't be easily vectorized.
    all_obstacle_geoms = objects.obstacle.geom_names + ['floor']
    contact_cost = 0
    for contact in phy.d.contact:
        geom_name1 = phy.m.geom(contact.geom1).name
        geom_name2 = phy.m.geom(contact.geom2).name
        if (geom_name1 in all_obstacle_geoms and geom_name2 in objects.val_collision_geom_names) or \
                (geom_name2 in all_obstacle_geoms and geom_name1 in objects.val_collision_geom_names) or \
                val_self_collision(geom_name1, geom_name2, objects):
            contact_cost += 1
    max_expected_contacts = 6.0
    contact_cost /= max_expected_contacts
    # clamp to be between 0 and 1, and more sensitive to just a few contacts
    contact_cost = min(np.power(contact_cost, hp['contact_exponent']), hp['max_contact_cost']) * hp['contact_cost']
    return contact_cost


def val_self_collision(geom_name1, geom_name2, objects: Objects):
    return geom_name1 in objects.val_self_collision_geom_names and geom_name2 in objects.val_self_collision_geom_names


def get_action_cost(joint_positions):
    action_cost = np.sum(np.abs(joint_positions[:, 1:] - joint_positions[:, :-1]), axis=-1)
    action_cost *= hp['action']
    return action_cost


def get_results_common(objects: Objects, phy: Physics):
    joint_indices_for_actuators = phy.m.actuator_trnid[:, 0]
    joint_positions = phy.d.qpos[joint_indices_for_actuators]
    contact_cost = get_contact_cost(phy, objects)
    left_tool_pos = phy.d.site('left_tool').xpos
    right_tool_pos = phy.d.site('right_tool').xpos
    return left_tool_pos, right_tool_pos, joint_positions, contact_cost


def get_rope_points(phy, rope_body_indices):
    rope_points = np.array([phy.d.xpos[rope_body_idx] for rope_body_idx in rope_body_indices])
    return rope_points
