import logging

import mujoco
import numpy as np

from mjregrasping.mujoco_visualizer import plot_sphere_rviz
from mjregrasping.body_with_children import BodyWithChildren

logger = logging.getLogger(f'rosout.{__name__}')


class MPPIGoal:

    def __init__(self, model, viz_pubs):
        self.model = model
        self.viz_pubs = viz_pubs

    def __getstate__(self):
        # Required for pickling
        state = self.__dict__.copy()
        del state["viz_pubs"]
        del state["model"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.viz_pubs = None
        self.model = None

    def cost(self, results):
        """
        Args:
            results: the output of get_results()

        Returns:
            matrix of costs [b, horizon]

        """
        raise NotImplementedError()

    def satisfied(self, data):
        raise NotImplementedError()

    def viz(self):
        raise NotImplementedError()

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

    def __init__(self, model, goal_point: np.array, goal_radius: float, gripper_idx: int, viz_pubs):
        super().__init__(model, viz_pubs)
        self.goal_point = goal_point
        self.goal_radius = goal_radius
        self.gripper_idx = gripper_idx

    def cost(self, results):
        """
        Args:
            results: the output of get_results()

        Returns:
            matrix of costs [b, horizon]

        """
        left_gripper_pos, right_gripper_pos = results
        gripper_point = self.choose_gripper_pos(left_gripper_pos, right_gripper_pos)
        dist_cost = np.linalg.norm(self.goal_point - gripper_point, axis=-1)[:, 1:]
        return dist_cost

    def satisfied(self, data):
        gripper_pos = self.choose_gripper_pos(data.site('left_tool').xpos, data.site('right_tool').xpos)
        distance = np.linalg.norm(self.goal_point - gripper_pos, axis=-1)
        return distance < self.goal_radius

    def viz(self):
        plot_sphere_rviz(self.viz_pubs.goal, position=self.goal_point, radius=self.goal_radius, frame_id='world',
                         color=[1, 0, 1, 0.5])
        if self.viz_pubs.tfw:
            self.viz_pubs.tfw.send_transform(translation=self.goal_point, quaternion=[0, 0, 0, 1],
                                             parent='world', child='gripper_point_goal')

    def get_results(self, model, data):
        """

        Args:
            model: mjModel
            data: mjData

        Returns: the result is any object or tuple of objects, and will be passed to cost()

        """
        return data.site('left_tool').xpos, data.site('right_tool').xpos

    def choose_gripper_pos(self, left_gripper_pos, right_gripper_pos):
        if self.gripper_idx == 0:
            gripper_point = left_gripper_pos
        elif self.gripper_idx == 1:
            gripper_point = right_gripper_pos
        else:
            raise ValueError(f"unknown gripper_idx {self.gripper_idx}")
        return gripper_point


    def tool_positions(self, results):
        return results


class ObjectPointGoal(MPPIGoal):

    def __init__(self, model, goal_point: np.array, goal_radius: float, body_idx: int, viz_pubs, rope, val, obstacle):
        super().__init__(model, viz_pubs)
        self.goal_point = goal_point
        self.body_idx = body_idx
        self.goal_radius = goal_radius
        self.rope = rope
        self.val = val
        self.obstacle = obstacle

    def cost(self, results):
        rope_points, joint_positions, left_tool_pos, right_tool_pos, is_grasping, contact_cost = results

        pred_rope_points = rope_points[:, 1:]
        pred_contact_cost = contact_cost[:, 1:]
        point_dist = self.min_dist_to_specified_point(pred_rope_points)
        gripper_points = np.stack([left_tool_pos, right_tool_pos], axis=-2)

        initial_point_dist = self.min_dist_to_specified_point(rope_points[:, 0])
        final_point_dist = point_dist[:, -1]
        near_threshold = 0.02
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
        # Cost the discouraged moveing the gripper away from the goal
        grasping_gripper_direction_cost = np.einsum('abc,abc->ab', gripper_direction_cost, pred_is_grasping)

        is_grasping0 = is_grasping[:, 0]  # grasp state cannot change within a rollout
        cannot_progress = np.logical_or(np.all(is_grasping0, axis=-1), np.all(~is_grasping0, axis=-1))
        cannot_progress_penalty = cannot_progress * 1000

        logger.debug(f"{any_points_useful=}")
        if any_points_useful:
            cost = point_dist
            # if self.spaces.debug.config.obj_point_goal:
            #     most_useful_point = np.argmax(initial_point_dist - final_point_dist)
            #     useful_points_traj = rope_points[most_useful_point]
            #     anim = RvizAnimationController(n_time_steps=useful_points_traj.shape[0], ns='time')
            #     while not anim.done:
            #         t = anim.t()
            #         r = useful_points_traj[t]
            #         plot_rope_rviz(self.viz_pubs.state_viz_pub, r, idx=0, label='useful')
            #         anim.step()
        else:
            cost = grasping_gripper_direction_cost
            # if self.spaces.debug.config.obj_point_goal:
            #     print("no points useful!")
            #     b_viz = np.topk(grasping_gripper_direction_cost.mean(axis=1), 1, largest=False).indices
            #     masked_gripper_direction = gripper_dir * is_grasping[:, None, :, None]
            #     grasping_gripper_direction = masked_gripper_direction.sum(axis=2).mean(axis=1)
            #     masked_grasping_gripper_positions = gripper_points * is_grasping[:, None, :, None]
            #     grasping_gripper_positions = masked_grasping_gripper_positions.sum(axis=2).mean(axis=1)
            #     plot_arrows_rviz(self.viz_pubs.state_viz_pub, numpify(specified_points[b_viz]),
            #                      numpify(specified_point_to_goal_dir[b_viz]), label='goal dir')
            #     plot_arrows_rviz(self.viz_pubs.state_viz_pub, numpify(grasping_gripper_positions[b_viz]),
            #                      5 * numpify(grasping_gripper_direction[b_viz]), label='gripper dir', color='white',
            #                      frame_id='val/base_body')

        cost += pred_contact_cost + np.expand_dims(cannot_progress_penalty, -1)
        if not any_points_useful:
            cost += 10
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

    def viz(self):
        plot_sphere_rviz(self.viz_pubs.goal, position=self.goal_point, radius=self.goal_radius, frame_id='world',
                         color=[1, 0, 1, 0.5])
        if self.viz_pubs.tfw:
            self.viz_pubs.tfw.send_transform(translation=self.goal_point, quaternion=[0, 0, 0, 1],
                                             parent='world', child='rope_point_goal')

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

        # doing the contact cost calculation here means we don't need to return the entire data.contact array,
        # which makes things simpler and possibly faster, since this operation can't be easily vectorized.
        contact_cost = 0
        for contact in data.contact:
            geom_name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom_name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            if (geom_name1 in self.obstacle.geom_names and geom_name2 in self.val.geom_names) or \
                    (geom_name2 in self.obstacle.geom_names and geom_name1 in self.val.geom_names):
                contact_cost += 1
        max_expected_contacts = 6.0
        contact_cost /= max_expected_contacts
        # clamp to be between 0 and 1, and more sensitive to just a few contacts
        # FIXME: hyperparameter should come from a config dict or something
        contact_cost = min(np.power(contact_cost, 0.5), 1.0)

        return rope_points, joint_positions, left_tool_pos, right_tool_pos, is_grasping, contact_cost

    def tool_positions(self, results):
        return results[2], results[3]
