import logging
from copy import copy

import mujoco
import numpy as np
import rerun as rr
from numpy.linalg import norm

from mjregrasping.body_with_children import Objects
from mjregrasping.params import Params
from mjregrasping.physics import Physics
from mjregrasping.viz import Viz

logger = logging.getLogger(f'rosout.{__name__}')


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
        self.viz.sphere(ns='goal', position=position, radius=radius, frame_id='world', color=[1, 0, 1, 0.5])
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


class GripperPointGoal(MPPIGoal):

    def __init__(self, goal_point: np.array, goal_radius: float, gripper_idx: int, objects: Objects,
                 viz: Viz):
        super().__init__(viz)
        self.goal_point = goal_point
        self.goal_radius = goal_radius
        self.gripper_idx = gripper_idx
        self.objects = objects
        self.p = self.viz.p

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
        action_cost = get_action_cost(joint_positions, self.p)

        cost = copy(pred_contact_cost)
        cost += dist_cost
        cost += action_cost

        rr.log_scalar('gripper_point_goal/pred_contact', pred_contact_cost.mean())
        rr.log_scalar('gripper_point_goal/action', action_cost.mean())
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
        contact_cost = get_contact_cost(phy, self.objects, self.p)
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
        self.p = self.viz.p
        self.initial_body_pos = None

    def cost(self, results):
        """
        Args:
            results: the output of get_results()

        Returns:
            matrix of costs [b, horizon]

        """
        left_gripper_pos, right_gripper_pos, joint_positions, body_pos, contact_cost = results
        pred_contact_cost = contact_cost[:, 1:]
        gripper_point = self.choose_gripper_pos(left_gripper_pos, right_gripper_pos)
        dist_cost = norm(body_pos - gripper_point, axis=-1)[:, 1:]
        action_cost = get_action_cost(joint_positions, self.p)

        if self.initial_body_pos is None:
            self.initial_body_pos = body_pos
        rope_motion_cost = norm(body_pos - self.initial_body_pos, axis=-1)[:, 1:]

        cost = copy(pred_contact_cost)
        cost += dist_cost
        cost += action_cost
        cost += rope_motion_cost

        rr.log_scalar('grasp_rope_goal/pred_contact', pred_contact_cost.mean())
        rr.log_scalar('grasp_rope_goal/rope_motion', rope_motion_cost.mean())
        rr.log_scalar('grasp_rope_goal/action', action_cost.mean())
        rr.log_scalar('grasp_rope_goal/dist', dist_cost.mean())

        return cost

    def satisfied(self, phy):
        body_pos = self.get_body_pos(phy.d)
        gripper_pos = self.choose_gripper_pos(phy.d.site('left_tool').xpos, phy.d.site('right_tool').xpos)
        distance = norm(body_pos - gripper_pos, axis=-1)
        return distance < self.goal_radius

    def viz_goal(self, phy):
        body_pos = self.get_body_pos(phy.d)
        self.viz_sphere(body_pos, self.goal_radius)

    def get_results(self, phy):
        """
        Returns: the result is any object or tuple of objects, and will be passed to cost()
        """
        joint_indices_for_actuators = phy.m.actuator_trnid[:, 0]
        joint_positions = phy.d.qpos[joint_indices_for_actuators]
        contact_cost = get_contact_cost(phy, self.objects, self.p)
        body_pos = self.get_body_pos(phy.d)
        return phy.d.site('left_tool').xpos, phy.d.site('right_tool').xpos, joint_positions, body_pos, contact_cost

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
        self.p = self.viz.p

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

        is_grasping0 = is_grasping[:, 0]  # grasp state cannot change within a rollout

        # Get the potential field gradient at the specified points
        gripper_dfield = self.dfield.get_costs(pred_gripper_points)  # [b, h, 2]
        gripper_dfield *= self.p.gripper_dfield
        grasping_gripper_dfield = np.sum(gripper_dfield * pred_is_grasping, -1)  # [b, horizon]

        cost = copy(pred_point_dist) * self.p.point_dist_weight

        cost += grasping_gripper_dfield

        cost += pred_contact_cost

        # Add cost for grippers that are not grasping
        # that encourages them to remain close to the rope
        # [b, horizon, n_rope_points, n_grippers]
        rope_gripper_dists = norm(pred_gripper_points[..., None, :, :] - pred_rope_points[..., None, :], axis=-1)
        pred_is_not_grasping = 1 - pred_is_grasping
        min_nongrasping_dists = np.sum(np.min(rope_gripper_dists, -2) * pred_is_not_grasping, -1)  # [b, horizon]
        min_nongrasping_dists = np.sqrt(np.maximum(min_nongrasping_dists - self.p.nongrasping_close, 0))

        min_nongrasping_cost = min_nongrasping_dists * self.p.min_nongrasping_rope_gripper_dists
        cost += min_nongrasping_cost

        # Add a cost that non-grasping grippers should try to return to a "home" position.
        # Home is assumed to be 0, so penalize the distance from 0.
        # FIXME: doesn't generalize, hard-coded for Val
        arm_gripper_matrix = np.zeros([20, 2])
        left_joint_indices = np.arange(2, 2 + 9)
        right_joint_indices = np.arange(11, 11 + 9)
        arm_gripper_matrix[left_joint_indices, 0] = 1
        arm_gripper_matrix[right_joint_indices, 1] = 1
        home_cost_joints = np.abs(pred_joint_positions)  # [b, horizon, n_joints]
        home_cost_grippers = home_cost_joints @ arm_gripper_matrix
        nongrasping_home_cost = np.sum(home_cost_grippers * pred_is_not_grasping, -1)  # [b, horizon]
        nongrasping_home_cost = nongrasping_home_cost * self.p.nongrasping_home
        cost += nongrasping_home_cost

        # Add an action cost
        action_cost = get_action_cost(joint_positions, self.p)
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
        rope_points = np.array([phy.d.geom_xpos[rope_geom_idx] for rope_geom_idx in self.objects.rope.geom_indices])
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

    def get_results(self, phy):
        left_tool_pos = phy.d.site('left_tool').xpos
        right_tool_pos = phy.d.site('right_tool').xpos
        rope_points = np.array([phy.d.geom_xpos[rope_geom_idx] for rope_geom_idx in self.objects.rope.geom_indices])
        joint_indices_for_actuators = phy.m.actuator_trnid[:, 0]
        joint_positions = phy.d.qpos[joint_indices_for_actuators]
        eq_indices = [
            mujoco.mj_name2id(phy.m, mujoco.mjtObj.mjOBJ_EQUALITY, 'left'),
            mujoco.mj_name2id(phy.m, mujoco.mjtObj.mjOBJ_EQUALITY, 'right'),
        ]
        is_grasping = phy.m.eq_active[eq_indices]

        contact_cost = get_contact_cost(phy, self.objects, self.p)

        return rope_points, joint_positions, left_tool_pos, right_tool_pos, is_grasping, contact_cost


def get_contact_cost(phy: Physics, objects: Objects, p: Params):
    # TODO: use SDF to compute near-contact cost to avoid getting too close
    # doing the contact cost calculation here means we don't need to return the entire data.contact array,
    # which makes things simpler and possibly faster, since this operation can't be easily vectorized.
    contact_cost = 0
    for contact in phy.d.contact:
        geom_name1 = phy.m.geom(contact.geom1).name
        geom_name2 = phy.m.geom(contact.geom2).name
        if (geom_name1 in objects.obstacle.geom_names and geom_name2 in objects.val.geom_names) or \
                (geom_name2 in objects.obstacle.geom_names and geom_name1 in objects.val.geom_names) or \
                val_self_collision(geom_name1, geom_name2, objects):
            contact_cost += 1
    max_expected_contacts = 6.0
    contact_cost /= max_expected_contacts
    # clamp to be between 0 and 1, and more sensitive to just a few contacts
    contact_cost = min(np.power(contact_cost, p.contact_exponent), p.max_contact_cost) * p.contact_cost
    return contact_cost


def val_self_collision(geom_name1, geom_name2, objects: Objects):
    return geom_name1 in objects.val_self_collision_geom_names and geom_name2 in objects.val_self_collision_geom_names


def get_action_cost(joint_positions, p: Params):
    action_cost = np.sum(np.abs(joint_positions[:, 1:] - joint_positions[:, :-1]), axis=-1)
    action_cost *= p.action
    return action_cost
