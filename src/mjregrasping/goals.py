import logging
from copy import deepcopy

import mujoco
import numpy as np
import rerun as rr
from numpy.linalg import norm

from mjregrasping.goal_funcs import get_results_common, get_rope_points, get_keypoint, get_finger_cost, get_action_cost, \
    get_tool_positions
from mjregrasping.grasp_state_utils import grasp_locations_to_indices_and_offsets_and_xpos, \
    grasp_locations_to_indices_and_offsets
from mjregrasping.grasping import get_is_grasping, get_finger_qs
from mjregrasping.ik import jacobian_ik_is_reachable
from mjregrasping.magnetic_fields import get_h_signature
from mjregrasping.geometry import pairwise_squared_distances
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

    def viz_ee_lines(self, tools_pos, idx: int, scale: float, color):
        for i, tool_pos in enumerate(np.moveaxis(tools_pos, 1, 0)):
            self.viz.lines(tool_pos, ns=f'ee_{i}', idx=idx, scale=scale, color=color)

    def viz_rope_lines(self, rope_pos, idx: int, scale: float, color):
        self.viz.lines(rope_pos, ns='rope', idx=idx, scale=scale, color=color)

    def get_results(self, phy):
        """
        Returns: the result is any object or tuple of objects, and will be passed to cost()
        The reason for this architecture is that returning the entire physics state is expensive, since it requires
        making a copy of it (because multiprocessing). So we only return the parts of the state that are needed for
        cost().
        """
        raise NotImplementedError()

    def viz_goal(self, phy):
        raise NotImplementedError()


class ObjectPointGoal(MPPIGoal):

    def __init__(self, goal_point: np.array, goal_radius: float, loc: float, viz: Viz):
        super().__init__(viz)
        self.goal_point = goal_point
        self.loc = loc
        self.goal_radius = goal_radius

    def get_results(self, phy):
        tools_pos, joint_positions, contact_cost, is_unstable = get_results_common()
        body_idx, offset = grasp_locations_to_indices_and_offsets(self.loc, phy.o.rope.body_indices)
        rope_points = get_rope_points(phy, phy.o.rope.body_indices)
        is_grasping = get_is_grasping(phy.m)

        keypoint = get_keypoint(phy, body_idx, offset)

        return result(rope_points, keypoint, joint_positions, tools_pos, is_grasping, contact_cost,
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
        body_idx, offset = grasp_locations_to_indices_and_offsets(self.loc, phy.o.rope.body_indices)
        keypoint = get_keypoint(phy, body_idx, offset)
        error = self.keypoint_dist_to_goal(keypoint).squeeze()
        return error < self.goal_radius

    def gripper_dists_to_goal(self, left_tool_pos, right_tool_pos):
        gripper_points = np.stack([left_tool_pos, right_tool_pos], axis=-2)
        gripper_distances = norm((gripper_points - self.goal_point[None, None, None]), axis=-1)
        return gripper_distances

    def keypoint_dist_to_goal(self, keypoint):
        return norm((keypoint - self.goal_point), axis=-1)

    def viz_result(self, result, idx: int, scale, color):
        tools_pos = as_float(result[2])
        keypoints = as_float(result[0])
        self.viz_ee_lines(tools_pos, idx, scale, color)
        self.viz_rope_lines(keypoints, idx, scale, color='y')

    def viz_goal(self, phy):
        self.viz_sphere(self.goal_point, self.goal_radius)


def get_regrasp_costs(finger_qs, is_grasping, regrasp_locs, regrasp_xpos, tools_pos, rope_points):
    """

    Args:
        finger_qs: Finger joint angle
        is_grasping:  Whether the gripper is grasping
        regrasp_locs:  Whether the gripper should grasp ∈ [0-1]
        regrasp_xpos: The 3d position in space corresponding to the regrasp_locs
        tools_pos: The current 3d position of the tool tips
        rope_points: The 3d position of all the rope points

    Returns:
        Costs for finger joint angles and tool tip positions

    """
    reach_is_grasping = regrasp_locs > 0
    regrasp_finger_cost = get_finger_cost(finger_qs, reach_is_grasping) * hp['finger_weight']
    regrasp_dists = norm(regrasp_xpos - tools_pos, axis=-1)
    needs_grasp = reach_is_grasping * (1 - is_grasping)
    regrasp_pos_cost = np.sum(regrasp_dists * needs_grasp, -1) * hp['regrasp_weight']

    dists = pairwise_squared_distances(regrasp_xpos, rope_points)
    min_dist = np.min(dists)
    regrasp_near_cost = min_dist * hp['regrasp_near_weight']
    return regrasp_finger_cost, regrasp_pos_cost, regrasp_near_cost


def ray_based_reachability(candidates_xpos, phy, tools_pos, max_d=0.7):
    # by having group 1 set to 0, we exclude the rope and grippers/fingers
    # FIXME: this reachability check is not accurate!
    include_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    is_reachable = np.zeros([len(tools_pos), len(candidates_xpos)], dtype=bool)
    for i, tool_pos in enumerate(tools_pos):
        for j, xpos in enumerate(candidates_xpos):
            out_geomids = np.array([-1], dtype=np.int32)
            candidate_to_tool = (tool_pos - xpos)
            # print(i, j, candidate_to_tool, out_geomids)
            if norm(candidate_to_tool) > max_d:
                # print("not reachable because the candidate is too far away")
                is_reachable[i, j] = False
                continue
            d = mujoco.mj_ray(phy.m, phy.d, xpos, candidate_to_tool, include_groups, 1, -1, out_geomids)
            if d > max_d or out_geomids[0] == -1:
                # print("reachable because either there was no collision, or the collision was far away")
                is_reachable[i, j] = True
            else:
                # print("not reachable because there was a collision and it was close")
                is_reachable[i, j] = False
    return is_reachable


def ik_based_reachability(candidates_xpos, phy, tools_pos):
    is_reachable = np.zeros([len(tools_pos), len(candidates_xpos)], dtype=bool)
    from time import perf_counter
    t0 = perf_counter()
    for i, tool_pos in enumerate(tools_pos):
        tool_body_idx = phy.m.body('left_finger_pad').id if i == 0 else phy.m.body('right_finger_pad').id
        for j, xpos in enumerate(candidates_xpos):
            is_reachable[i, j] = jacobian_ik_is_reachable(phy, tool_body_idx, xpos, pos_tol=0.03)
    print(f'dt: {perf_counter() - t0:.4f}')
    return is_reachable


class RegraspGoal(MPPIGoal):

    def __init__(self, op_goal, skeletons, grasp_goal_radius, viz: Viz):
        super().__init__(viz)
        self.op_goal = op_goal
        self.skeletons = skeletons
        self.grasp_goal_radius = grasp_goal_radius
        self.n_g = hp['n_g']
        self.reach_rng = np.random.RandomState(0)
        self.homotopy_rng = np.random.RandomState(1)

    def satisfied(self, phy):
        return self.op_goal.satisfied(phy)

    def viz_goal(self, phy):
        self.op_goal.viz_goal(phy)

    def get_results(self, phy: Physics):
        # Create goals that makes the closest point on the rope the intended goal
        # For create a result tuple for each gripper, based on gripper_action
        # The only case where we want to make a gripper result/cost is when we are not currently grasping
        # but when gripper_action is 1, meaning change the grasp state.
        tools_pos, _, contact_cost, is_unstable = get_results_common(phy)
        is_grasping = get_is_grasping(phy.m)
        rope_points = get_rope_points(phy, phy.o.rope.body_indices)
        finger_qs = get_finger_qs(phy)
        op_goal_body_idx, op_goal_offset = grasp_locations_to_indices_and_offsets(self.op_goal.loc,
                                                                                  phy.o.rope.body_indices)
        keypoint = get_keypoint(phy, op_goal_body_idx, op_goal_offset)

        _, _, reach_xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy, self.reach_locs)
        _, _, homotopy_xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy, self.homotopy_locs)

        return result(tools_pos, contact_cost, is_grasping, is_unstable, rope_points, keypoint, finger_qs, reach_xpos,
                      homotopy_xpos)

    def gen_grasp_for_new_homotopy(self, is_grasping, phy, tools_pos, attach_pos):
        robot_base_pos = phy.d.body('val_base').xpos
        h0 = []
        # from mjregrasping.rerun_visualizer import log_skeletons
        # log_skeletons(self.skeletons, color=(0, 255, 0, 255), timeless=True)
        for tool_pos in tools_pos:
            path = np.stack([robot_base_pos, tool_pos, attach_pos, robot_base_pos], axis=0)
            h = get_h_signature(path, self.skeletons)
            h0.append(h)
            # rr.log_line_strip('path/0', path, ext={'hs': str(h0)})
        h0 = np.reshape(np.array(h0) * is_grasping[:, None], -1)
        # Uniformly randomly sample a new grasp
        # reject if it's the same as the current grasp
        # reject if it's not reachable
        # check if it's H-signature is different
        allowable_is_grasping = np.array([[0, 1], [1, 0], [1, 1]])
        while True:
            homotopy_is_grasping = allowable_is_grasping[self.homotopy_rng.randint(0, 3)]
            homotopy_locs = self.homotopy_rng.uniform(0, 1, 2)
            homotopy_locs = -1 * (1 - homotopy_is_grasping) + homotopy_locs * homotopy_is_grasping

            # homotopy_is_grasping = np.array([1, 1])
            # homotopy_locs = np.array([0.85, 0.95])
            _, _, homotopy_xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy, homotopy_locs)

            hi = []
            for i, pos in enumerate(homotopy_xpos):
                path = np.stack([robot_base_pos, pos, attach_pos, robot_base_pos], axis=0)
                h = get_h_signature(path, self.skeletons)
                hi.append(h)
                # rr.log_line_strip(f'path/{i + 1}', path, ext={'hs': str(hi)})
            hi = np.reshape(np.array(hi) * homotopy_is_grasping[:, None], -1)

            reachable_matrix = ray_based_reachability(homotopy_xpos, phy, tools_pos)
            homotopy_not_grasping = np.logical_not(homotopy_is_grasping)
            reachable_or_not_grasping = np.logical_or(np.diagonal(reachable_matrix), homotopy_not_grasping)
            valid = np.all(reachable_or_not_grasping) and np.any(h0 != hi) and np.any(homotopy_is_grasping)
            if valid:
                break
        return homotopy_locs

    def gen_grasp_for_controllability(self, is_grasping, phy, tools_pos):
        not_grasping = 1 - is_grasping
        p_reach_gripper = softmax(not_grasping, 0.1)
        reach_is_grasping = self.reach_rng.binomial(1, p_reach_gripper)
        candidates_reach_locs = np.linspace(0, 1, 10)
        candidates_bodies, candidates_offsets, candidates_xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy,
                                                                                                                 candidates_reach_locs)
        is_reachable = ray_based_reachability(candidates_xpos, phy, tools_pos)
        geodesics_costs = np.square(candidates_reach_locs - self.op_goal.loc)
        combined_costs = geodesics_costs + 1000 * (1 - is_reachable)
        best_idx = np.argmin(combined_costs, axis=-1)
        reach_locs = candidates_reach_locs[best_idx]
        reach_locs = reach_locs * reach_is_grasping + -1 * (1 - reach_is_grasping)
        return reach_locs

    def cost(self, results, is_grasping0):
        (tools_pos, contact_cost, is_grasping, is_unstable, rope_points, keypoint, finger_qs, reach_xpos,
         homotopy_xpos) = as_floats(results)

        keypoint_dist = norm(keypoint - self.op_goal.goal_point, axis=-1)

        unstable_cost = is_unstable * hp['unstable_weight']

        goal_cost = keypoint_dist * hp['goal_weight']

        maintain_grasps_cost = get_finger_cost(finger_qs, is_grasping0) * hp['finger_weight']

        # NOTE: reading class variables from multiple processes without any protection!
        # Reach costs
        reach_finger_cost, reach_pos_cost, reach_near_cost = get_regrasp_costs(finger_qs, is_grasping,
                                                                               self.reach_locs, reach_xpos, tools_pos,
                                                                               rope_points)

        # Homotopy costs
        homotopy_finger_cost, homotopy_pos_cost, homotopy_near_cost = get_regrasp_costs(finger_qs, is_grasping,
                                                                                        self.homotopy_locs,
                                                                                        homotopy_xpos, tools_pos,
                                                                                        rope_points)

        # TODO: MAB should choose these weights
        w_goal = self.viz.p.config['w_goal']
        w_reach = self.viz.p.config['w_reach']
        w_homotopy = self.viz.p.config['w_homotopy']
        # w_grasp_nearest = self.viz.p.config['w_grasp_nearest']
        return (
            contact_cost,
            unstable_cost,
            w_goal * goal_cost,
            w_goal * maintain_grasps_cost,
            w_reach * reach_finger_cost,
            w_reach * reach_pos_cost,
            w_reach * reach_near_cost,
            w_homotopy * homotopy_finger_cost,
            w_homotopy * homotopy_pos_cost,
            w_homotopy * homotopy_near_cost,
        )

    def cost_names(self):
        return [
            "contact",
            "unstable",
            "goal",
            "maintain_grasps_cost",
            "reach_finger_cost",
            "reach_pos_cost",
            "reach_near_cost",
            "homotopy_finger_cost",
            "homotopy_pos_cost",
            "homotopy_near_cost",
        ]

    def viz_result(self, result, idx: int, scale, color):
        tools_pos = as_float(result[0])
        keypoints = as_float(result[5])
        t0 = 0
        reach_xpos = as_float(result[7])[t0]
        homotopy_xpos = as_float(result[8])[t0]
        self.viz_ee_lines(tools_pos, idx, scale, color)
        self.viz_rope_lines(keypoints, idx, scale, color='y')

        self.viz.sphere('left_reach_xpos', reach_xpos[0], radius=0.02, color='b', frame_id='world', idx=0)
        self.viz.sphere('right_reach_xpos', reach_xpos[1], radius=0.02, color='b', frame_id='world', idx=0)
        self.viz.sphere('left_homotopy_xpos', homotopy_xpos[0], radius=0.02, color='g', frame_id='world', idx=0)
        self.viz.sphere('right_homotopy_xpos', homotopy_xpos[1], radius=0.02, color='g', frame_id='world', idx=0)

    def recompute_candidates(self, phy, attach_pos):
        from time import perf_counter
        t0 = perf_counter()
        tools_pos = get_tool_positions(phy)
        is_grasping = get_is_grasping(phy.m)

        # Reachability planner
        # - minimize geodesic distance to the keypoint (loc)
        # - subject to reachability constraint, which might be hard but should probably involve collision-free IK?
        self.reach_locs = self.gen_grasp_for_controllability(is_grasping, phy, tools_pos)

        # Homotopy planner
        # Find a new grasp configuration that results in a new homotopy class,
        # and satisfies reachability constraints
        # We can start by trying rejection sampling?
        if attach_pos is None:
            self.homotopy_locs = np.array([-1, -1])
        else:
            self.homotopy_locs = self.gen_grasp_for_new_homotopy(is_grasping, phy, tools_pos, attach_pos)

        print(f'Recompute candidates: {perf_counter() - t0:.4f}')
        return self.reach_locs, self.homotopy_locs
