from copy import deepcopy
from time import perf_counter
from typing import Dict

import numpy as np
import pysdf_tools
from numpy.linalg import norm

from mjregrasping.goal_funcs import get_results_common, get_rope_points, get_keypoint, \
    get_nongrasping_rope_contact_cost, get_regrasp_costs
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets, grasp_locations_to_xpos
from mjregrasping.grasp_strategies import get_strategy
from mjregrasping.grasping import get_is_grasping, get_grasp_locs, get_finger_qs
from mjregrasping.grasp_and_settle import grasp_and_settle
from mjregrasping.homotopy_checker import get_full_h_signature_from_phy
from mjregrasping.teleport_to_plan import teleport_to_end_of_plan
from mjregrasping.homotopy_utils import skeleton_field_dir
from mjregrasping.my_transforms import angle_between
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.rrt import GraspRRT
from mjregrasping.viz import Viz
from moveit_msgs.msg import MotionPlanResponse


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
    # self.viz.arrow('bfield_dir', rope_points[0, -1], 0.5 * bfield_dirs[0, -1], cm.Reds(angle_cost[0] / np.pi))
    # self.viz.arrow('delta', rope_points[0, -1], rope_deltas[0, -1], cm.Reds(angle_cost[0] / np.pi))
    angle_cost = sum(angle_cost)
    return angle_cost


def get_pull_through_cost(loc, rope_points, goal_dir):
    rope_deltas = rope_points[1:] - rope_points[:-1]  # [t-1, n, 3]
    angle_cost = angle_between(rope_deltas, goal_dir)
    # weight by the geodesic distance from each rope point to the loc we're threading through
    w = np.exp(-hp['thread_geodesic_w'] * np.abs(np.linspace(0, 1, rope_points.shape[1]) - loc))
    angle_cost = angle_cost @ w * hp['angle_cost_weight']
    # self.viz.arrow('bfield_dir', rope_points[0, -1], 0.5 * bfield_dirs[0, -1], cm.Reds(angle_cost[0] / np.pi))
    # self.viz.arrow('delta', rope_points[0, -1], rope_deltas[0, -1], cm.Reds(angle_cost[0] / np.pi))
    angle_cost = sum(angle_cost)
    return angle_cost


def get_next_xpos_sdf_cost(sdf, next_xpos, next_locs):
    next_is_grasping = (next_locs != -1)
    sdf_dists = np.zeros(next_xpos.shape[:2])
    for t, xpos_t in enumerate(next_xpos):
        for i, xpos_ti in enumerate(xpos_t):
            dist = sdf.GetValueByCoordinates(*xpos_ti)[0]
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
        self.viz.tf(translation=position, quat_xyzw=[0, 0, 0, 1], parent='world', child='goal')

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

    def get_grasp_locs(self):
        return self.locs

    def set_grasp_locs(self, grasp_locs):
        self.locs = grasp_locs


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

        grasp_finger_cost, grasp_pos_cost, grasp_near_cost = get_regrasp_costs(finger_qs, is_grasping,
                                                                               current_locs, grasp_locs,
                                                                               grasp_xpos, tools_pos, rope_points)

        gripper_to_goal_cost = np.sum(norm(tools_pos - self.goal_point, axis=-1) * is_grasping, axis=-1)
        gripper_to_goal_cost = gripper_to_goal_cost * hp['gripper_to_goal_weight']

        smoothness_cost = get_smoothness_cost(u_sample)

        contact_cost = sum(contact_cost)

        grasp_finger_cost = sum(grasp_finger_cost)
        grasp_pos_cost = sum(grasp_pos_cost)
        grasp_near_cost = sum(grasp_near_cost)
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
            grasp_near_cost,
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
            "keypooint",
            "grasp_finger",
            "grasp_pos",
            "grasp_near",
            "nongrasping_rope_contact",
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


class ThreadingGoal(ObjectPointGoalBase):

    def __init__(self, grasp_goal: GraspLocsGoal, skeletons: Dict, skeleton_name, loc: float, next_tool_name: str,
                 next_locs, next_h,
                 grasp_rrt: GraspRRT, sdf: pysdf_tools.SignedDistanceField, viz: Viz):
        self.skel = skeletons[skeleton_name]
        goal_point = np.mean(self.skel, axis=0)
        super().__init__(goal_point, loc, viz)

        self.skeletons = skeletons
        self.goal_dir = skeleton_field_dir(self.skel, self.goal_point[None])[0] * 0.01
        self.next_tool_name = next_tool_name
        self.next_locs = next_locs
        self.next_h = next_h
        self.grasp_rrt = grasp_rrt
        self.sdf = sdf
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
        next_xpos = grasp_locations_to_xpos(phy, self.next_locs)

        return result(tools_pos, contact_cost, is_grasping, current_locs, is_unstable, rope_points, keypoint,
                      finger_qs, grasp_locs, grasp_xpos, next_xpos, joint_positions, nongrasping_rope_contact_cost)

    def viz_goal(self, phy: Physics):
        self.viz.arrow('goal_dir', self.goal_point, self.goal_dir, 'g')
        xpos = grasp_locations_to_xpos(phy, [self.loc])[0]
        self.viz.sphere('current_loc', xpos, 0.015, 'g')

    def costs(self, results, u_sample):
        (tools_pos, contact_cost, is_grasping, current_locs, is_unstable, rope_points, keypoint, finger_qs,
         grasp_locs, grasp_xpos, next_xpos, joint_positions, nongrasping_rope_contact_cost) = as_floats(results)

        unstable_cost = is_unstable * hp['unstable_weight']

        nongrasping_rope_contact_cost = sum(nongrasping_rope_contact_cost * hp['nongrasping_rope_contact_weight'])

        grasp_finger_cost, grasp_pos_cost, grasp_near_cost = get_regrasp_costs(finger_qs, is_grasping,
                                                                               current_locs, grasp_locs,
                                                                               grasp_xpos, tools_pos, rope_points)
        grasp_finger_cost = sum(grasp_finger_cost)
        grasp_pos_cost = sum(grasp_pos_cost)
        grasp_near_cost = sum(grasp_near_cost)

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

        sdf_cost = get_next_xpos_sdf_cost(self.sdf, next_xpos, self.next_locs)

        return (
            contact_cost,
            unstable_cost,
            angle_cost,
            keypoint_cost,
            sdf_cost,
            grasp_finger_cost,
            grasp_pos_cost,
            grasp_near_cost,
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
            "threading_angle",
            "threading_keypoint",
            "threading_sdf",
            "grasp_finger",
            "grasp_pos",
            "grasp_near",
            "nongrasping_rope_contact",
            "gripper_to_goal",
            "ever_not_grasping",
            "smoothness",
        ]

    def satisfied(self, phy: Physics):
        # See if we can grasp the next loc and if we can, and it's the right homotopy, then we're done
        res, scene_msg = self.plan_to_next_locs(phy)
        if res is None:
            return False
        else:
            satisfied = self.satisfied_from_res(phy, res)
            return satisfied

    def satisfied_from_res(self, phy: Physics, res: MotionPlanResponse):
        if res is None:
            return False

        phy_plan = phy.copy_all()
        teleport_to_end_of_plan(phy_plan, res)
        grasp_and_settle(phy_plan, self.next_locs, viz=None, is_planning=True)

        h, _ = get_full_h_signature_from_phy(self.skeletons, phy_plan,
                                             collapse_empty_gripper_cycles=False,
                                             gripper_ids_in_h_signature=True)
        satisfied = h == self.next_h
        return satisfied

    def plan_to_next_locs(self, phy: Physics):
        current_locs = get_grasp_locs(phy)
        strategy = get_strategy(current_locs, self.next_locs)

        t0 = perf_counter()
        res, scene_msg = self.grasp_rrt.plan(phy, strategy, self.next_locs, viz=self.viz, allowed_planning_time=1.0)
        plan_found = res.error_code.val == res.error_code.SUCCESS
        print(f'dt: {perf_counter() - t0:.4f}, {plan_found=}')
        if not plan_found:
            return None, None

        # self.viz.rviz.viz_scene(scene_msg)
        # self.grasp_rrt.display_result(self.viz, res, scene_msg)
        return res, scene_msg


class PullThroughGoal(ThreadingGoal):

    def costs(self, results, u_sample):
        (tools_pos, contact_cost, is_grasping, current_locs, is_unstable, rope_points, keypoint, finger_qs,
         grasp_locs, grasp_xpos, next_xpos, joint_positions, nongrasping_rope_contact_cost) = as_floats(results)

        unstable_cost = is_unstable * hp['unstable_weight']

        nongrasping_rope_contact_cost = nongrasping_rope_contact_cost * hp['nongrasping_rope_contact_weight']

        grasp_finger_cost, grasp_pos_cost, grasp_near_cost = get_regrasp_costs(finger_qs, is_grasping,
                                                                               current_locs, grasp_locs,
                                                                               grasp_xpos, tools_pos, rope_points)

        gripper_to_goal_cost = np.sum(norm(tools_pos - self.goal_point, axis=-1) * is_grasping, axis=-1)
        gripper_to_goal_cost = gripper_to_goal_cost * hp['gripper_to_goal_weight']

        smoothness_cost = get_smoothness_cost(u_sample)

        contact_cost = sum(contact_cost)
        unstable_cost = sum(unstable_cost)
        grasp_finger_cost = sum(grasp_finger_cost)
        grasp_pos_cost = sum(grasp_pos_cost)
        grasp_near_cost = sum(grasp_near_cost)
        nongrasping_rope_contact_cost = sum(nongrasping_rope_contact_cost)
        gripper_to_goal_cost = sum(gripper_to_goal_cost)

        no_gripper_grasping = np.any(np.all(np.logical_not(is_grasping), axis=-1), axis=-1)
        ever_not_grasping_cost = no_gripper_grasping * hp['ever_not_grasping_weight']

        keypoint_dist = self.keypoint_dist_to_goal(keypoint)[1:]
        keypoint_cost = sum(keypoint_dist * hp['keypoint_weight'])

        angle_cost = get_pull_through_cost(self.loc, rope_points, self.goal_dir)

        sdf_cost = get_next_xpos_sdf_cost(self.sdf, next_xpos, self.next_locs)

        return (
            contact_cost,
            unstable_cost,
            angle_cost,
            keypoint_cost,
            sdf_cost,
            grasp_finger_cost,
            grasp_pos_cost,
            grasp_near_cost,
            nongrasping_rope_contact_cost,
            gripper_to_goal_cost,
            ever_not_grasping_cost,
            smoothness_cost,
        )
