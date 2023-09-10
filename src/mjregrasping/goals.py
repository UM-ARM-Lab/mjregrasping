from copy import deepcopy
from typing import Dict, Optional, List

import numpy as np
from numpy.linalg import norm

from mjregrasping.geometry import pairwise_squared_distances
from mjregrasping.goal_funcs import get_results_common, get_rope_points, get_keypoint, \
    get_nongrasping_rope_contact_cost, get_regrasp_costs
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets, grasp_locations_to_xpos
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_is_grasping, get_grasp_locs, get_finger_qs
from mjregrasping.homotopy_utils import skeleton_field_dir
from mjregrasping.my_transforms import angle_between
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.viz import Viz
from tf.transformations import quaternion_from_matrix
from visualization_msgs.msg import MarkerArray, Marker


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


def get_angle_cost(skel, loc, rope_points):
    rope_deltas = rope_points[1:] - rope_points[:-1]  # [t-1, n, 3]
    bfield_dirs_flat = skeleton_field_dir(skel, rope_points[:-1].reshape(-1, 3))
    bfield_dirs = bfield_dirs_flat.reshape(rope_deltas.shape)  # [t-1, n, 3]
    angle_cost = angle_between(rope_deltas, bfield_dirs)
    # weight by the geodesic distance from each rope point to the loc we're threading through
    w = np.exp(-hp['thread_geodesic_w'] * np.abs(np.linspace(0, 1, rope_points.shape[1]) - loc))
    angle_cost = angle_cost @ w * hp['angle_cost_weight']
    angle_cost = sum(angle_cost)
    if np.any(~np.isfinite(angle_cost)):
        return 100
    return angle_cost


def get_next_xpos_sdf_cost(sdf, next_xpos, next_locs):
    next_is_grasping = (next_locs != -1)
    sdf_dists = np.zeros(next_xpos.shape[:2])
    for t, xpos_t in enumerate(next_xpos):
        for i, xpos_ti in enumerate(xpos_t):
            dist = np.clip(sdf.GetValueByCoordinates(*xpos_ti)[0], -1, 1)
            sdf_dists[t, i] = dist
    sdf_cost = np.sum(np.exp(-sdf_dists) * hp['next_xpos_sdf_weight'] * next_is_grasping, axis=1)
    sdf_cost = sum(sdf_cost)
    return sdf_cost


def get_smoothness_cost(u_sample):
    du = (u_sample[1:] - u_sample[:-1])
    smoothness_costs = norm(du, axis=-1)
    smoothness_cost = smoothness_costs * hp['smoothness_weight']
    smoothness_cost = sum(smoothness_cost)
    return smoothness_cost


class MPPIGoal:

    def __init__(self, viz: Viz):
        self.viz = viz

    def satisfied(self, phy: Physics):
        raise NotImplementedError()

    def viz_sphere(self, position, radius):
        self.viz.sphere(ns='goal', position=position, radius=radius, color=[1, 0, 1, 0.5], idx=0, frame_id='world')

    def viz_result(self, phy: Physics, result, idx: int, scale, color):
        raise NotImplementedError()

    def viz_ee_lines(self, tools_pos, idx: int, scale: float, color):
        for i, tool_pos in enumerate(np.moveaxis(tools_pos, 1, 0)):
            self.viz.lines(tool_pos, ns=f'ee_{i}', idx=idx, scale=scale, color=color)

    def viz_rope_lines(self, rope_pos, idx: int, scale: float, color):
        self.viz.lines(rope_pos, ns='rope', idx=idx, scale=scale, color=color)

    def get_results(self, phy: Physics):
        """
        Returns: the result is any object or tuple of objects, and will be passed to cost()
        The reason for this architecture is that returning the entire physics state is expensive, since it requires
        making a copy of it (because multiprocessing). So we only return the parts of the state that are needed for
        cost().
        """
        raise NotImplementedError()

    def viz_goal(self, phy: Physics):
        raise NotImplementedError()


class ObjectPointGoalBase(MPPIGoal):
    def __init__(self, goal_point: np.array, loc: float, viz: Viz):
        super().__init__(viz)
        self.goal_point = goal_point
        self.loc = loc

    def keypoint_dist_to_goal(self, keypoint):
        return norm((keypoint - self.goal_point), axis=-1)

    def viz_result(self, phy: Physics, result, idx: int, scale, color):
        tools_pos = as_float(result[0])
        keypoints = as_float(result[6])
        self.viz_ee_lines(tools_pos, idx, scale, color)
        self.viz_rope_lines(keypoints, idx, scale, color='y')


class GraspLocsGoal:
    """ This is just a wrapper around the grasp locations, so that we can pass it to RegraspGoal """

    def __init__(self, current_locs):
        self.locs = current_locs
        self.history = [current_locs]

    def get_grasp_locs(self):
        return self.locs

    def set_grasp_locs(self, grasp_locs, is_planning=False):
        self.locs = grasp_locs
        if not is_planning:
            self.history.append(grasp_locs)


class ObjectPointGoal(ObjectPointGoalBase):

    def __init__(self, grasp_goal: GraspLocsGoal, goal_point: np.array, goal_radius: float, loc: float, viz: Viz):
        super().__init__(goal_point, loc, viz)
        self.goal_radius = goal_radius
        self.grasp_goal = grasp_goal

    def get_results(self, phy: Physics):
        tools_pos, joint_positions, contact_cost, is_unstable = get_results_common(phy)
        is_grasping = get_is_grasping(phy)
        rope_points = get_rope_points(phy)
        finger_qs = get_finger_qs(phy)
        op_goal_body_idx, op_goal_offset = grasp_locations_to_indices_and_offsets(self.loc, phy)
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

        unstable_cost = sum(is_unstable * hp['unstable_weight'])

        nongrasping_rope_contact_cost = nongrasping_rope_contact_cost * hp['nongrasping_rope_contact_weight']

        grasp_finger_cost, grasp_pos_cost = get_regrasp_costs(finger_qs, is_grasping, current_locs, grasp_locs,
                                                              grasp_xpos, tools_pos)

        gripper_to_goal_cost = np.sum(norm(tools_pos - self.goal_point, axis=-1) * is_grasping, axis=-1)
        gripper_to_goal_cost = gripper_to_goal_cost * hp['gripper_to_goal_weight']

        smoothness_cost = get_smoothness_cost(u_sample)

        contact_cost = sum(contact_cost)

        nongrasping_rope_dist_cost = get_nongrasping_near_cost(is_grasping, rope_points, tools_pos)

        grasp_finger_cost = sum(grasp_finger_cost)
        grasp_pos_cost = sum(grasp_pos_cost)
        nongrasping_rope_contact_cost = sum(nongrasping_rope_contact_cost)
        gripper_to_goal_cost = sum(gripper_to_goal_cost)

        no_gripper_grasping = np.any(np.all(np.logical_not(is_grasping), axis=-1), axis=-1)
        ever_not_grasping_cost = no_gripper_grasping * hp['ever_not_grasping_weight']

        keypoint_dist = self.keypoint_dist_to_goal(keypoint)[1:]
        keypoint_cost = sum(keypoint_dist * hp['keypoint_weight'])

        return (
            contact_cost,
            unstable_cost,
            keypoint_cost,
            grasp_finger_cost,
            grasp_pos_cost,
            nongrasping_rope_contact_cost,
            nongrasping_rope_dist_cost,
            gripper_to_goal_cost,
            ever_not_grasping_cost,
            smoothness_cost,
        )

    @staticmethod
    def cost_names():
        return [
            "contact",
            "unstable",
            "keypooint",
            "grasp_finger",
            "grasp_pos",
            "nongrasping_rope_contact",
            "nongrasping_rope_dist",
            "gripper_to_goal",
            "ever_not_grasping",
            "smoothness",
        ]

    def satisfied(self, phy: Physics):
        body_idx, offset = grasp_locations_to_indices_and_offsets(self.loc, phy)
        keypoint = get_keypoint(phy, body_idx, offset)
        error = self.keypoint_dist_to_goal(keypoint).squeeze()
        return error < self.goal_radius

    def viz_goal(self, phy: Physics):
        self.viz_sphere(self.goal_point, self.goal_radius)


def perturb_locs(strategy, locs):
    next_locs = locs + np.random.randn() * 0.03
    next_locs = np.clip(next_locs, 0, 1)
    next_locs = np.where([s in [Strategies.NEW_GRASP, Strategies.MOVE] for s in strategy], next_locs, locs)
    return next_locs


def get_nongrasping_near_cost(is_grasping, rope_points, tools_pos):
    min_dists_to_rope = np.min(np.abs(pairwise_squared_distances(rope_points, tools_pos) - 0.10), axis=1)
    nongrasping_rope_dist_cost = np.sum(min_dists_to_rope * np.logical_not(is_grasping), axis=-1)
    nongrasping_rope_dist_cost = sum(nongrasping_rope_dist_cost) * hp['nongrasping_rope_dist_weight']
    return nongrasping_rope_dist_cost


class ThreadingGoal(ObjectPointGoalBase):

    def __init__(self, grasp_goal: GraspLocsGoal, skeletons: Dict, skeleton_names: List[str], loc: float, viz: Viz):
        """

        Args:
            grasp_goal:
            skeletons:
            skeleton_names: skeletons to use for threading in order
            loc: This is analogous to the first reference point in Weifu's method
            viz:
        """
        self.skeleton_names = skeleton_names
        self.skel = skeletons[skeleton_names[-1]]
        goal_point = np.mean(self.skel[:4], axis=0)
        super().__init__(goal_point, loc, viz)

        self.goal_dir = skeleton_field_dir(self.skel, self.goal_point[None])[0] * 0.01
        self.grasp_goal = grasp_goal

    def get_results(self, phy: Physics):
        tools_pos, joint_positions, contact_cost, is_unstable = get_results_common(phy)
        is_grasping = get_is_grasping(phy)
        rope_points = get_rope_points(phy)
        finger_qs = get_finger_qs(phy)
        op_goal_body_idx, op_goal_offset = grasp_locations_to_indices_and_offsets(self.loc, phy)
        keypoint = get_keypoint(phy, op_goal_body_idx, op_goal_offset)

        current_locs = get_grasp_locs(phy)

        grasp_locs = self.grasp_goal.get_grasp_locs()

        nongrasping_rope_contact_cost = get_nongrasping_rope_contact_cost(phy, grasp_locs)

        grasp_xpos = grasp_locations_to_xpos(phy, grasp_locs)

        return result(tools_pos, contact_cost, is_grasping, current_locs, is_unstable, rope_points, keypoint,
                      finger_qs, grasp_locs, grasp_xpos, joint_positions, nongrasping_rope_contact_cost)

    def viz_goal(self, phy: Physics):
        self.viz.arrow('goal_dir', self.goal_point, self.goal_dir, 'g')
        xpos = grasp_locations_to_xpos(phy, [self.loc])[0]
        self.viz.sphere('current_loc', xpos, 0.015, 'g')

    def costs(self, results, u_sample):
        (tools_pos, contact_cost, is_grasping, current_locs, is_unstable, rope_points, keypoint, finger_qs,
         grasp_locs, grasp_xpos, joint_positions, nongrasping_rope_contact_cost) = as_floats(results)

        unstable_cost = is_unstable * hp['unstable_weight']

        nongrasping_rope_contact_cost = sum(nongrasping_rope_contact_cost * hp['nongrasping_rope_contact_weight'])

        nongrasping_rope_dist_cost = get_nongrasping_near_cost(is_grasping, rope_points, tools_pos)

        grasp_finger_cost, grasp_pos_cost = get_regrasp_costs(finger_qs, is_grasping, current_locs, grasp_locs,
                                                              grasp_xpos, tools_pos)
        grasp_finger_cost = sum(grasp_finger_cost)
        grasp_pos_cost = sum(grasp_pos_cost)

        gripper_to_goal_cost = np.sum(norm(tools_pos - self.goal_point, axis=-1) * is_grasping, axis=-1)
        gripper_to_goal_cost = gripper_to_goal_cost * hp['gripper_to_goal_weight']
        gripper_to_goal_cost = sum(gripper_to_goal_cost)

        smoothness_cost = get_smoothness_cost(u_sample)

        contact_cost = sum(contact_cost)
        unstable_cost = sum(unstable_cost)

        no_gripper_grasping = np.any(np.all(np.logical_not(is_grasping), axis=-1), axis=-1)
        ever_not_grasping_cost = no_gripper_grasping * hp['ever_not_grasping_weight']

        keypoint_dist = self.keypoint_dist_to_goal(keypoint)[1:]
        keypoint_cost = sum(keypoint_dist * hp['keypoint_weight'])

        angle_cost = get_angle_cost(self.skel, self.loc, rope_points)

        torso_cost = sum(np.abs(joint_positions[:, 1] - 0.2) * hp['torso_weight'])

        return (
            contact_cost,
            unstable_cost,
            angle_cost,
            keypoint_cost,
            torso_cost,
            grasp_finger_cost,
            grasp_pos_cost,
            nongrasping_rope_contact_cost,
            nongrasping_rope_dist_cost,
            gripper_to_goal_cost,
            ever_not_grasping_cost,
            smoothness_cost,
        )

    @staticmethod
    def cost_names():
        return [
            "contact",
            "unstable",
            "threading_angle",
            "threading_keypoint",
            "torso_cost",
            "grasp_finger",
            "grasp_pos",
            "nongrasping_rope_contact",
            "nongrasping_rope_dist",
            "gripper_to_goal",
            "ever_not_grasping",
            "smoothness",
        ]

    def satisfied(self, phy: Physics, disc_center, disc_normal, disc_rad):
        # NOTE: always using 1 here means regardless of which loc we're trying to push through the loop,
        #  check whether the tip is penetrating the disc
        satisfied = penetrating_disc_approximation(disc_center, disc_normal, disc_rad, phy, 1.0, self.viz)
        return satisfied


def penetrating_disc_approximation(disc_center, disc_normal, disc_rad, phy: Physics, loc: float, viz: Optional[Viz]):
    disc_normal = disc_normal / norm(disc_normal)
    xpos = grasp_locations_to_xpos(phy, [loc])[0]
    # project xpos into the place of the disc, defined by the disc center and normal
    # https://math.stackexchange.com/questions/96061/how-to-project-a-point-onto-a-plane-in-3d
    projected_xpos = xpos - np.dot(xpos - disc_center, disc_normal) * disc_normal
    # check if the projected xpos is within the disc
    dist = norm(projected_xpos - disc_center)
    # also check if the origin xpos is on the positive half of the disc plane
    satisfied = dist < disc_rad and np.dot(xpos - disc_center, disc_normal) > 0
    if viz:
        viz.sphere('projected_xpos', projected_xpos, 0.001, 'g' if satisfied else 'r')
        disc_marker_msg = Marker()
        disc_marker_msg.type = Marker.CYLINDER
        disc_marker_msg.ns = 'disc'
        disc_marker_msg.header.frame_id = 'world'
        disc_marker_msg.pose.position.x = disc_center[0]
        disc_marker_msg.pose.position.y = disc_center[1]
        disc_marker_msg.pose.position.z = disc_center[2]
        # create a quaternion such that the z axis is the normal
        z = disc_normal
        x = np.cross(z, np.random.randn(3))
        x /= norm(x)
        y = np.cross(z, x)
        rot_mat = np.eye(4)
        rot_mat[:3, 0] = x
        rot_mat[:3, 1] = y
        rot_mat[:3, 2] = z
        quat = quaternion_from_matrix(rot_mat)
        disc_marker_msg.pose.orientation.x = quat[0]
        disc_marker_msg.pose.orientation.y = quat[1]
        disc_marker_msg.pose.orientation.z = quat[2]
        disc_marker_msg.pose.orientation.w = quat[3]
        disc_marker_msg.scale.x = disc_rad * 2
        disc_marker_msg.scale.y = disc_rad * 2
        disc_marker_msg.scale.z = 0.001
        disc_marker_msg.color.g = 1
        disc_marker_msg.color.a = 0.2

        disc_msg = MarkerArray()
        disc_msg.markers.append(disc_marker_msg)
        viz.markers_pub.publish(disc_msg)
    return satisfied


def point_goal_from_geom(grasp_goal: GraspLocsGoal, phy: Physics, geom: str, loc: float, viz: Viz):
    goal_point = phy.d.geom(geom).xpos
    goal_radius = phy.m.geom(geom).size[0] - 0.01
    return ObjectPointGoal(grasp_goal, goal_point, goal_radius, loc, viz)
