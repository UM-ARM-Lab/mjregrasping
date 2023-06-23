import logging
from copy import deepcopy

import mujoco
import numpy as np
import rerun as rr
from matplotlib import cm
from numpy.linalg import norm

from mjregrasping.body_with_children import Objects
from mjregrasping.geometry import pairwise_squared_distances
from mjregrasping.goal_funcs import get_action_cost, get_contact_cost, get_results_common, get_rope_points, \
    compute_threading_dir, get_keypoint
from mjregrasping.grasp_state_utils import grasp_indices_to_locations, grasp_locations_to_indices_and_offsets
from mjregrasping.grasping import get_is_grasping, get_finger_qs
from mjregrasping.my_transforms import angle_between
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.rviz import plot_spheres_rviz
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
        left_gripper_pos, right_gripper_pos, joint_positions, contact_cost, is_unstable = as_floats(results)
        pred_contact_cost = contact_cost[:, 1:]
        gripper_point = self.choose_gripper_pos(left_gripper_pos, right_gripper_pos)
        dist_cost = norm(self.goal_point - gripper_point, axis=-1)[:, 1:]
        action_cost = get_action_cost(joint_positions)

        cost = pred_contact_cost + dist_cost + action_cost

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
        return result(phy.d.site('left_tool').xpos, phy.d.site('right_tool').xpos, joint_positions, contact_cost)

    def viz_result(self, result, idx: int, scale, color):
        left_tool_pos = as_float(result[0])
        right_tool_pos = as_float(result[1])
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


class CombinedGoal(ObjectPointGoal):

    def __init__(self, goal_point: np.array, goal_radius: float, body_idx: int, objects, viz: Viz):
        super().__init__(goal_point, goal_radius, body_idx, objects, viz)
        self.rope_body_indices = np.array(self.objects.rope.body_indices)
        self.n_b = len(self.rope_body_indices)
        self.n_g = hp['n_g']
        # cached intermediate result used by the MPC algorithm in regrasp_mpc.py
        self.grasp_costs = None

    def get_results(self, phy):
        left_tool_pos, right_tool_pos, joint_positions, contact_cost, is_unstable = get_results_common(self.objects,
                                                                                                       phy)
        rope_points = np.array([phy.d.xpos[rope_body_idx] for rope_body_idx in self.objects.rope.body_indices])
        eq_active = np.concatenate([phy.m.eq("left").active, phy.m.eq("right").active])
        eq_obj2id = np.concatenate([phy.m.eq("left").obj2id, phy.m.eq("right").obj2id])

        return result(left_tool_pos, right_tool_pos, joint_positions, contact_cost, rope_points, eq_active, eq_obj2id,
                      is_unstable)

    def cost(self, results):
        left_pos, right_pos, joint_positions, contact_cost, rope_points, eq_active, eq_obj2id, is_unstable = as_floats(
            results)
        pred_contact_cost = contact_cost[:, 1:]  # skip t=0
        pred_joint_positions = joint_positions[:, 1:]

        raise NotImplementedError("fixme!")
        keypoint_dist = self.keypoint_dist_to_goal(rope_points)
        pred_point_dist = keypoint_dist[:, 1:] * hp['point_dist_weight']

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

        action_cost = get_action_cost(pred_joint_positions)

        unstable_cost = is_unstable[:, 1:] * 1e3

        goal_costs = pred_point_dist + pred_grasping_pull_cost

        grasp_w, weighted_grasp_costs = self.compute_weighted_grasp_costs(eq_active, eq_obj2id, left_pos, right_pos,
                                                                          rope_points)
        total_costs = goal_costs + weighted_grasp_costs + action_cost + pred_contact_cost + unstable_cost

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
        left_pos = as_float(result[0])
        right_pos = as_float(result[1])
        rope_pos = as_float(result[4])[:, self.body_idx]
        self.viz_ee_lines(left_pos, right_pos, idx, scale, color)
        self.viz_rope_lines(rope_pos, idx, scale, color='y')


class CombinedThreadingGoal(CombinedGoal):
    def __init__(self, goal_point: np.array, goal_dir, goal_radius: float, body_idx: int, objects, viz: Viz):
        super().__init__(goal_point, goal_radius, body_idx, objects, viz)
        self.goal_dir = goal_dir

    def cost(self, results):
        left_pos, right_pos, joint_positions, contact_cost, rope_points, eq_active, eq_obj2id, is_unstable = as_floats(
            results)
        pred_joint_positions = joint_positions[:, 1:]
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

        raise NotImplementedError("fixme!")
        point_dist = self.keypoint_dist_to_goal(rope_points)
        pred_point_dist = point_dist[:, 1:] * hp['point_dist_weight']

        action_cost = get_action_cost(pred_joint_positions)

        unstable_cost = is_unstable[:, 1:] * hp['unstable_weight']

        goal_costs = thread_dir_costs + pred_thread_orient_costs

        grasp_w, weighted_grasp_costs = self.compute_weighted_grasp_costs(eq_active, eq_obj2id, left_pos, right_pos,
                                                                          rope_points)
        total_costs = goal_costs + weighted_grasp_costs + action_cost + pred_contact_cost + unstable_cost

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
        leftgripper_q, rightgripper_q = get_finger_qs(phy)
        keypoint = get_keypoint(phy, self.op_goal.body_idx, self.op_goal.offset)

        return result(left_tool_pos, right_tool_pos, contact_cost, is_grasping, is_unstable, rope_points, keypoint,
                      leftgripper_q,
                      rightgripper_q)

    def cost(self, results, exploration_weight):
        left_tool_pos, right_tool_pos, contact_cost, is_grasping, is_unstable, rope_points, keypoint, leftgripper_q, rightgripper_q = as_floats(
            results)

        keypoint_dist = norm(keypoint - self.op_goal.goal_point, axis=-1)

        unstable_cost = is_unstable * hp['unstable_weight']

        goal_cost = keypoint_dist * hp['point_dist_weight']

        # Compute the minimum distance between each gripper and each point on the rope and min over rope points
        # then mask by needs_grasp
        tool_pos = np.stack([left_tool_pos, right_tool_pos], axis=0)  # [n_g, 3]
        dists = pairwise_squared_distances(tool_pos, rope_points)  # [n_g, n_p]
        min_dists = dists.min(axis=-1)  # [n_g]
        not_grasping = np.logical_not(is_grasping)
        min_dists_nongrasping = np.sum(min_dists * not_grasping, -1) * hp['grasp_weight'] * exploration_weight

        # Encourage grasping the point we care about bringing to the goal
        controllability = norm(tool_pos - keypoint[None], axis=-1)
        controllability_nongrasping = np.sum(controllability * not_grasping, -1)
        controllability_cost = controllability_nongrasping * hp['controllability_weight'] * exploration_weight

        finger_error = abs(leftgripper_q - hp['desired_finger_q']) + abs(rightgripper_q - hp['desired_finger_q'])
        finger_cost = finger_error * hp['finger_weight']
        # TODO: what should the release cost be?
        # finger_qs = np.stack([leftgripper_q, rightgripper_q], axis=-1)
        # release_grasping = np.dot(is_grasping, 1 - finger_qs)  # 1 is fully open
        # release_cost = release_grasping * exploration_weight
        release_cost = 0

        return contact_cost, goal_cost, unstable_cost, finger_cost, min_dists_nongrasping, controllability_cost, release_cost

    def move_cost_names(self):
        return ["contact", "goal", "unstable", "finger", "min_dists_nongrasping", "controllability", "release"]

    def viz_result(self, result, idx: int, scale, color):
        left_tool_pos = as_float(result[0])
        right_tool_pos = as_float(result[1])
        keypoints = as_float(result[6])
        self.viz_ee_lines(left_tool_pos, right_tool_pos, idx, scale, color)
        self.viz_rope_lines(keypoints, idx, scale, color='y')
