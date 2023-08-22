from copy import deepcopy
from time import perf_counter
from typing import Dict

import numpy as np
from numpy.linalg import norm

from mjregrasping.goal_funcs import get_results_common, get_rope_points, get_keypoint
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets, grasp_locations_to_xpos
from mjregrasping.grasp_strategies import get_strategy
from mjregrasping.grasping import get_is_grasping, get_grasp_locs
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
        tools_pos = as_float(result[2])
        keypoints = as_float(result[0])
        self.viz_ee_lines(tools_pos, idx, scale, color)
        self.viz_rope_lines(keypoints, idx, scale, color='y')


class ObjectPointGoal(ObjectPointGoalBase):

    def __init__(self, goal_point: np.array, goal_radius: float, loc: float, viz: Viz):
        super().__init__(goal_point, loc, viz)
        self.goal_radius = goal_radius

    def get_results(self, phy: Physics):
        tools_pos, joint_positions, contact_cost, is_unstable = get_results_common(phy)
        body_idx, offset = grasp_locations_to_indices_and_offsets(self.loc, phy)
        rope_points = get_rope_points(phy)
        is_grasping = get_is_grasping(phy)

        keypoint = get_keypoint(phy, body_idx, offset)

        return result(rope_points, keypoint, joint_positions, tools_pos, is_grasping, contact_cost,
                      is_unstable)

    def cost(self, rope_points, keypoint):
        return self.keypoint_dist_to_goal(keypoint)

    def satisfied(self, phy: Physics):
        body_idx, offset = grasp_locations_to_indices_and_offsets(self.loc, phy)
        keypoint = get_keypoint(phy, body_idx, offset)
        error = self.keypoint_dist_to_goal(keypoint).squeeze()
        return error < self.goal_radius

    def viz_goal(self, phy: Physics):
        self.viz_sphere(self.goal_point, self.goal_radius)


class ThreadingGoal(ObjectPointGoalBase):

    def __init__(self, skeletons: Dict, skeleton_name, loc: float, next_tool_name: str, next_locs, next_h,
                 grasp_rrt: GraspRRT, viz: Viz):
        self.skel = skeletons[skeleton_name]
        goal_point = np.mean(self.skel, axis=0)
        super().__init__(goal_point, loc, viz)

        self.skeletons = skeletons
        self.goal_dir = skeleton_field_dir(self.skel, self.goal_point[None])[0] * 0.01
        self.next_tool_name = next_tool_name
        self.next_locs = next_locs
        self.next_h = next_h
        self.grasp_rrt = grasp_rrt

    def viz_goal(self, phy: Physics):
        self.viz.arrow('goal_dir', self.goal_point, self.goal_dir, 'g')
        xpos = grasp_locations_to_xpos(phy, [self.loc])[0]
        self.viz.sphere('current_loc', xpos, 0.015, 'g')
        next_xpos = grasp_locations_to_xpos(phy, self.next_locs)
        for tool_name, next_xpos in zip(phy.o.rd.tool_sites, next_xpos):
            self.viz.sphere(f'next_loc_{tool_name}', next_xpos, 0.015, (1, 0, 1, 0.2))

    def cost(self, rope_points, keypoint):
        rope_deltas = rope_points[1:] - rope_points[:-1]  # [t-1, n, 3]
        bfield_dirs_flat = skeleton_field_dir(self.skel, rope_points[:-1].reshape(-1, 3))
        bfield_dirs = bfield_dirs_flat.reshape(rope_deltas.shape)  # [t-1, n, 3]
        angle_cost = angle_between(rope_deltas, bfield_dirs)
        # weight by the geodesic distance from each rope point to self.loc
        w = np.exp(-hp['thread_geodesic_w'] * np.abs(np.linspace(0, 1, rope_points.shape[1]) - self.loc))
        angle_cost = angle_cost @ w * hp['angle_cost_weight']
        # self.viz.arrow('bfield_dir', rope_points[0, -1], 0.5 * bfield_dirs[0, -1], cm.Reds(angle_cost[0] / np.pi))
        # self.viz.arrow('delta', rope_points[0, -1], rope_deltas[0, -1], cm.Reds(angle_cost[0] / np.pi))

        # skip the first keypoint dist cost, since it's constant across all samples, and we need to angle_cost shape
        keypoint_dist = self.keypoint_dist_to_goal(keypoint)[1:]
        return angle_cost + keypoint_dist

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
        res, scene_msg = self.grasp_rrt.plan(phy, strategy, self.next_locs, viz=None, allowed_planning_time=1.0)
        # res, scene_msg = self.grasp_rrt.plan(phy, strategy, self.next_locs, viz=self.viz, allowed_planning_time=1.0)
        plan_found = res.error_code.val == res.error_code.SUCCESS
        print(f'dt: {perf_counter() - t0:.4f}, {plan_found=}')
        if not plan_found:
            return None, None

        # self.viz.rviz.viz_scene(scene_msg)
        # self.grasp_rrt.display_result(res, scene_msg)
        return res, scene_msg


class GraspLocsGoal:
    """ This is just a wrapper around the grasp locations, so that we can pass it to RegraspGoal """

    def __init__(self, current_locs):
        self.locs = current_locs

    def get_grasp_locs(self):
        return self.locs

    def set_grasp_locs(self, grasp_locs):
        self.locs = grasp_locs
