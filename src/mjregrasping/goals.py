from copy import deepcopy
import matplotlib.cm as cm
from pathlib import Path

import numpy as np
from numpy.linalg import norm

from mjregrasping.goal_funcs import get_results_common, get_rope_points, get_keypoint
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets
from mjregrasping.grasping import get_is_grasping
from mjregrasping.homotopy_utils import skeleton_field_dir, get_h_signature
from mjregrasping.my_transforms import angle_between
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.viz import Viz


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


class ObjectPointGoal(MPPIGoal):

    def __init__(self, goal_point: np.array, goal_radius: float, loc: float, viz: Viz):
        super().__init__(viz)
        self.goal_point = goal_point
        self.loc = loc
        self.goal_radius = goal_radius

    def get_results(self, phy: Physics):
        tools_pos, joint_positions, contact_cost, is_unstable = get_results_common()
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

    def keypoint_dist_to_goal(self, keypoint):
        return norm((keypoint - self.goal_point), axis=-1)

    def viz_result(self, phy: Physics, result, idx: int, scale, color):
        tools_pos = as_float(result[2])
        keypoints = as_float(result[0])
        self.viz_ee_lines(tools_pos, idx, scale, color)
        self.viz_rope_lines(keypoints, idx, scale, color='y')

    def viz_goal(self, phy: Physics):
        self.viz_sphere(self.goal_point, self.goal_radius)


class ThreadingGoal(ObjectPointGoal):

    def __init__(self, skeleton: np.ndarray, demo_path: Path, loc: float, viz: Viz):
        goal_point = np.mean(skeleton, axis=0)
        goal_radius = 0.05  # TODO: deduce this from the skeleton
        super().__init__(goal_point, goal_radius, loc, viz)

        self.skel = skeleton
        self.goal_dir = skeleton_field_dir(skeleton, self.goal_point[None])[0] * 0.01
        self.demo_path = demo_path

    def viz_goal(self, phy: Physics):
        super().viz_goal(phy)
        self.viz.arrow('goal_dir', self.goal_point, self.goal_dir, 'g')

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
        rope_points = get_rope_points(phy)
        rope_loop = np.concatenate([rope_points, [
            rope_points[-1] + np.array([0, 0, -1]),
            rope_points[0] + np.array([0, 0, -1]),
            rope_points[0]
        ]], axis=0)
        # self.viz.lines(rope_loop, ns='rope_loop', idx=0, scale=0.01, color='b')
        h = get_h_signature(rope_loop, {'obs': self.skel})[0]
        through = (h == 1)
        return through
