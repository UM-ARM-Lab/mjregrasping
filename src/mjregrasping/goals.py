import logging
from copy import copy

import mujoco
import numpy as np
import rerun as rr

from mjregrasping.body_with_children import BodyWithChildren
from mjregrasping.params import Params
from mjregrasping.viz import Viz

logger = logging.getLogger(f'rosout.{__name__}')


class MPPIGoal:

    def __init__(self, model, visualizer: Viz):
        self.model = model
        self.visualizer = visualizer

    def __getstate__(self):
        # Required for pickling
        state = self.__dict__.copy()
        del state["visualizer"]
        del state["model"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.visualizer = None
        self.model = None

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

    def satisfied(self, data):
        raise NotImplementedError()

    def viz_sphere(self, position, radius):
        self.visualizer.sphere(ns='goal', position=position, radius=radius, frame_id='world', color=[1, 0, 1, 0.5])
        self.visualizer.tf(translation=position, quat_xyzw=[0, 0, 0, 1], parent='world', child='goal')

    def get_results(self, model, data):
        """

        Args:
            model: mjModel
            data: mjData

        Returns: the result is any object or tuple of objects, and will be passed to cost()

        """
        raise NotImplementedError()

    def tool_positions(self, results):
        raise NotImplementedError()


class GripperPointGoal(MPPIGoal):

    def __init__(self, model, goal_point: np.array, goal_radius: float, gripper_idx: int, visualizer: Viz, rope, val,
                 obstacle,
                 p: Params):
        super().__init__(model, visualizer)
        self.goal_point = goal_point
        self.goal_radius = goal_radius
        self.gripper_idx = gripper_idx
        self.rope = rope
        self.val = val
        self.obstacle = obstacle
        self.p = p

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
        dist_cost = np.linalg.norm(self.goal_point - gripper_point, axis=-1)[:, 1:]
        action_cost = get_action_cost(joint_positions, self.p)

        cost = copy(pred_contact_cost)
        cost += dist_cost
        cost += action_cost

        rr.log_scalar('costs/pred_contact', pred_contact_cost.mean())
        rr.log_scalar('costs/action', action_cost.mean())
        rr.log_scalar('costs/dist', dist_cost.mean())

        return cost

    def satisfied(self, data):
        gripper_pos = self.choose_gripper_pos(data.site('left_tool').xpos, data.site('right_tool').xpos)
        distance = np.linalg.norm(self.goal_point - gripper_pos, axis=-1)
        return distance < self.goal_radius

    def viz(self, d):
        self.viz_sphere(self.goal_point, self.goal_radius)

    def get_results(self, model, data):
        """

        Args:
            model: mjModel
            data: mjData

        Returns: the result is any object or tuple of objects, and will be passed to cost()

        """
        joint_indices_for_actuators = model.actuator_trnid[:, 0]
        joint_positions = data.qpos[joint_indices_for_actuators]
        contact_cost = get_contact_cost(self.model, data, self.obstacle, self.val, self.p)
        return data.site('left_tool').xpos, data.site('right_tool').xpos, joint_positions, contact_cost

    def choose_gripper_pos(self, left_gripper_pos, right_gripper_pos):
        if self.gripper_idx == 0:
            gripper_point = left_gripper_pos
        elif self.gripper_idx == 1:
            gripper_point = right_gripper_pos
        else:
            raise ValueError(f"unknown gripper_idx {self.gripper_idx}")
        return gripper_point

    def tool_positions(self, results):
        return results[0], results[1]


class GraspBodyGoal(MPPIGoal):

    def __init__(self, model, body_id_to_grasp: int, goal_radius: float, gripper_idx: int, visualizer: Viz, rope,
                 val, obstacle, p: Params):
        super().__init__(model, visualizer)
        self.body_id_to_grasp = body_id_to_grasp
        self.goal_radius = goal_radius
        self.gripper_idx = gripper_idx
        self.rope = rope
        self.val = val
        self.obstacle = obstacle
        self.p = p

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
        dist_cost = np.linalg.norm(body_pos - gripper_point, axis=-1)[:, 1:]
        action_cost = get_action_cost(joint_positions, self.p)

        cost = copy(pred_contact_cost)
        cost += dist_cost
        cost += action_cost

        rr.log_scalar('costs/pred_contact', pred_contact_cost.mean())
        rr.log_scalar('costs/action', action_cost.mean())
        rr.log_scalar('costs/dist', dist_cost.mean())

        return cost

    def satisfied(self, data):
        body_pos = self.get_body_pos(data)
        gripper_pos = self.choose_gripper_pos(data.site('left_tool').xpos, data.site('right_tool').xpos)
        distance = np.linalg.norm(body_pos - gripper_pos, axis=-1)
        return distance < self.goal_radius

    def viz(self, d):
        body_pos = self.get_body_pos(d)
        self.viz_sphere(body_pos, self.goal_radius)

    def get_results(self, model, data):
        """

        Args:
            model: mjModel
            data: mjData

        Returns: the result is any object or tuple of objects, and will be passed to cost()

        """
        joint_indices_for_actuators = model.actuator_trnid[:, 0]
        joint_positions = data.qpos[joint_indices_for_actuators]
        contact_cost = get_contact_cost(self.model, data, self.obstacle, self.val, self.p)
        body_pos = self.get_body_pos(data)
        return data.site('left_tool').xpos, data.site('right_tool').xpos, joint_positions, body_pos, contact_cost

    def get_body_pos(self, data):
        body_pos = data.xpos[self.body_id_to_grasp]
        return body_pos

    def choose_gripper_pos(self, left_gripper_pos, right_gripper_pos):
        if self.gripper_idx == 0:
            gripper_point = left_gripper_pos
        elif self.gripper_idx == 1:
            gripper_point = right_gripper_pos
        else:
            raise ValueError(f"unknown gripper_idx {self.gripper_idx}")
        return gripper_point

    def tool_positions(self, results):
        return results[0], results[1]


class ObjectPointGoal(MPPIGoal):

    def __init__(self, model, goal_point: np.array, goal_radius: float, body_idx: int, viz_pubs, rope, val, obstacle,
                 p: Params):
        super().__init__(model, viz_pubs)
        self.goal_point = goal_point
        self.body_idx = body_idx
        self.goal_radius = goal_radius
        self.rope = rope
        self.val = val
        self.obstacle = obstacle
        self.p = p

    def cost(self, results):
        rope_points, joint_positions, left_tool_pos, right_tool_pos, is_grasping, contact_cost = results

        pred_rope_points = rope_points[:, 1:]
        pred_contact_cost = contact_cost[:, 1:]
        point_dist = self.min_dist_to_specified_point(pred_rope_points)
        gripper_points = np.stack([left_tool_pos, right_tool_pos], axis=-2)
        pred_gripper_points = gripper_points[:, 1:]

        initial_point_dist = self.min_dist_to_specified_point(rope_points[:, 0])
        final_point_dist = point_dist[:, -1]
        near_threshold = self.p.near_threshold
        points_can_progress = (final_point_dist + near_threshold) < initial_point_dist
        points_near_goal = initial_point_dist < (2 * near_threshold)
        points_useful = np.logical_or(points_can_progress, points_near_goal)
        any_points_useful = points_useful.any()
        gripper_dir = (gripper_points[:, 1:] - gripper_points[:, :-1])  # [b, horizon, 2, 3]
        pred_is_grasping = is_grasping[:, 1:]  # skip t=0
        specified_points = pred_rope_points[..., 0, self.body_idx, :]
        specified_point_to_goal_dir = (self.goal_point - specified_points)  # [b, 3]
        # negative dot product between:
        # 1. the direction from the specified point to the goal
        # 2. the direction the gripper is moving
        gripper_direction_cost = -np.einsum('abcd,ad->abc', gripper_dir, specified_point_to_goal_dir)  # [b, horizon, 2]
        # since the dot product is bounded from -1 to 1,
        # add 1 and divide by 2 to make it a little easier to compare to the other costs
        gripper_direction_cost = (gripper_direction_cost + 1) / 2
        # Cost the discouraged moving the gripper away from the goal
        grasping_gripper_direction_cost = np.einsum('abc,abc->ab', gripper_direction_cost, pred_is_grasping)
        grasping_gripper_direction_cost *= self.p.gripper_dir

        is_grasping0 = is_grasping[:, 0]  # grasp state cannot change within a rollout
        cannot_progress = np.logical_or(np.all(is_grasping0, axis=-1), np.all(np.logical_not(is_grasping0), axis=-1))
        cannot_progress_penalty = cannot_progress * self.p.cannot_progress

        logger.debug(f"{any_points_useful=}")
        if any_points_useful:
            cost = copy(point_dist)
        else:
            # copy because we want to be able to plot the original value of this later
            cost = copy(grasping_gripper_direction_cost)

        cost += np.expand_dims(cannot_progress_penalty, -1)

        cost += pred_contact_cost

        if not any_points_useful:
            no_points_useful_cost = self.p.no_points_useful
            cost += no_points_useful_cost

        # add cost for grippers that are not grasping
        # that encourages them to remain close to the rope
        # [b, horizon, n_rope_points, n_grippers]
        rope_gripper_dists = np.linalg.norm(
            np.expand_dims(pred_gripper_points, -3) - np.expand_dims(pred_rope_points, -2),
            axis=-1)
        pred_is_not_grasping = 1 - pred_is_grasping
        min_nongrasping_dists = np.sum(np.min(rope_gripper_dists, -2) * pred_is_not_grasping, -1)  # [b, horizon]
        min_nongrasping_dists = np.sqrt(np.maximum(min_nongrasping_dists - self.p.nongrasping_close, 0))

        min_nongrasping_cost = min_nongrasping_dists * self.p.min_nongrasping_rope_gripper_dists
        cost += min_nongrasping_cost

        action_cost = get_action_cost(joint_positions, self.p)
        cost += action_cost

        rr.log_scalar('costs/any_points_useful', any_points_useful)
        rr.log_scalar('costs/cannot_progress', cannot_progress)
        rr.log_scalar('costs/points', point_dist.mean())
        rr.log_scalar('costs/gripper_dir', grasping_gripper_direction_cost.mean())
        rr.log_scalar('costs/pred_contact', pred_contact_cost.mean())
        rr.log_scalar('costs/min_nongrasping', min_nongrasping_cost.mean())
        rr.log_scalar('costs/action', action_cost.mean())

        return cost  # [b, horizon]

    def satisfied(self, data):
        rope_points = np.array([data.geom_xpos[rope_geom_idx] for rope_geom_idx in self.rope.geom_indices])
        error = self.min_dist_to_specified_point(rope_points).squeeze()
        return error < self.goal_radius

    def gripper_dists_to_specified_point(self, left_tool_pos, right_tool_pos):
        gripper_points = np.stack([left_tool_pos, right_tool_pos], axis=-2)
        gripper_distances = np.linalg.norm((gripper_points - self.goal_point[None, None, None]), axis=-1)
        return gripper_distances

    def min_dist_to_specified_point(self, points):
        return self.min_dist_from_points_to_specified_point(points[..., self.body_idx, :])

    def min_dist_from_points_to_specified_point(self, points):
        return np.linalg.norm((points - self.goal_point), axis=-1)

    def viz(self, d):
        self.viz_sphere(self.goal_point, self.goal_radius)

    def get_results(self, model, data):
        left_tool_pos = data.site('left_tool').xpos
        right_tool_pos = data.site('right_tool').xpos
        rope_points = np.array([data.geom_xpos[rope_geom_idx] for rope_geom_idx in self.rope.geom_indices])
        joint_indices_for_actuators = model.actuator_trnid[:, 0]
        joint_positions = data.qpos[joint_indices_for_actuators]
        eq_indices = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'left'),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'right'),
        ]
        is_grasping = model.eq_active[eq_indices]

        contact_cost = get_contact_cost(self.model, data, self.obstacle, self.val, self.p)

        return rope_points, joint_positions, left_tool_pos, right_tool_pos, is_grasping, contact_cost

    def tool_positions(self, results):
        return results[2], results[3]


def get_contact_cost(model, data, obstacle: BodyWithChildren, val: BodyWithChildren, p: Params):
    # doing the contact cost calculation here means we don't need to return the entire data.contact array,
    # which makes things simpler and possibly faster, since this operation can't be easily vectorized.
    contact_cost = 0
    for contact in data.contact:
        geom_name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom_name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        if (geom_name1 in obstacle.geom_names and geom_name2 in val.geom_names) or \
                (geom_name2 in obstacle.geom_names and geom_name1 in val.geom_names):
            contact_cost += 1
    max_expected_contacts = 6.0
    contact_cost /= max_expected_contacts
    # clamp to be between 0 and 1, and more sensitive to just a few contacts
    contact_cost = min(np.power(contact_cost, p.contact_exponent),
                       p.max_contact_cost)
    return contact_cost


def get_action_cost(joint_positions, p: Params):
    action_cost = np.sum(np.abs(joint_positions[:, 1:] - joint_positions[:, :-1]), axis=-1)
    action_cost *= p.action
    return action_cost
