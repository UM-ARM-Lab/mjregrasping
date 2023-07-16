import numpy as np
# noinspection PyUnresolvedReferences
from mjregrasping.cfg import ParamsConfig
from numpy.linalg import norm

from mjregrasping.goal_funcs import get_results_common, get_rope_points, get_keypoint, get_regrasp_costs
from mjregrasping.goals import MPPIGoal, result, as_floats, as_float
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets, \
    grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.grasping import get_is_grasping, get_finger_qs, get_grasp_locs
from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.sdf_collision_checker import SDFCollisionChecker
from mjregrasping.viz import Viz


class RegraspGoal(MPPIGoal):

    def __init__(self, op_goal, skeletons, sdf, grasp_goal_radius, viz: Viz):
        super().__init__(viz)
        self.op_goal = op_goal
        self.skeletons = skeletons
        self.grasp_goal_radius = grasp_goal_radius
        self.cc = SDFCollisionChecker(sdf)
        self.homotopy_gen = HomotopyRegraspPlanner(op_goal, skeletons, self.cc)
        self.arm = ParamsConfig.Params_Goal
        self.current_locs = None

    def satisfied(self, phy: Physics):
        return self.op_goal.satisfied(phy)

    def viz_goal(self, phy: Physics):
        self.op_goal.viz_goal(phy)

    def update_arm(self, phy, arm):
        # TODO: MAB should choose this
        if arm != self.arm:
            self.current_locs = get_grasp_locs(phy)
        if self.current_locs is None:
            self.current_locs = get_grasp_locs(phy)
        self.arm = arm

    def get_results(self, phy: Physics):
        # Create goals that makes the closest point on the rope the intended goal
        # For create a result tuple for each gripper, based on gripper_action
        # The only case where we want to make a gripper result/cost is when we are not currently grasping
        # but when gripper_action is 1, meaning change the grasp state.
        tools_pos, _, contact_cost, is_unstable = get_results_common(phy)
        is_grasping = get_is_grasping(phy)
        rope_points = get_rope_points(phy)
        finger_qs = get_finger_qs(phy)
        op_goal_body_idx, op_goal_offset = grasp_locations_to_indices_and_offsets(self.op_goal.loc, phy)
        keypoint = get_keypoint(phy, op_goal_body_idx, op_goal_offset)

        current_locs = get_grasp_locs(phy)

        if self.arm == ParamsConfig.Params_Goal:
            grasp_locs = self.current_locs
        elif self.arm == ParamsConfig.Params_Homotopy:
            grasp_locs = self.homotopy_locs
        else:
            raise NotImplementedError(self.arm)

        _, _, grasp_xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy, grasp_locs)

        return result(tools_pos, contact_cost, is_grasping, current_locs, is_unstable, rope_points, keypoint,
                      finger_qs, grasp_locs, grasp_xpos)

    def cost(self, results):
        (tools_pos, contact_cost, is_grasping, current_locs, is_unstable, rope_points, keypoint, finger_qs,
         grasp_locs, grasp_xpos) = as_floats(results)
        keypoint_dist = norm(keypoint - self.op_goal.goal_point, axis=-1)

        unstable_cost = is_unstable * hp['unstable_weight']

        goal_cost = keypoint_dist * hp['goal_weight']

        # NOTE: reading class variables from multiple processes without any protection!
        grasp_finger_cost, grasp_pos_cost, grasp_near_cost = get_regrasp_costs(finger_qs, is_grasping,
                                                                               current_locs, grasp_locs,
                                                                               grasp_xpos, tools_pos, rope_points)

        # penalize distance of q's from 0
        home_cost = np.sum(np.abs(finger_qs)) * hp['home_weight']

        w_goal = 1 if self.arm == ParamsConfig.Params_Goal else 0
        return (
            contact_cost,
            unstable_cost,
            w_goal * goal_cost,
            grasp_finger_cost,
            grasp_pos_cost,
            grasp_near_cost,
            home_cost,
        )

    @staticmethod
    def cost_names():
        return [
            "contact",
            "unstable",
            "goal",
            "grasp_finger",
            "grasp_pos",
            "grasp_near",
            "home"
        ]

    def viz_result(self, phy: Physics, result, idx: int, scale, color):
        tools_pos = as_float(result[0])
        keypoints = as_float(result[6])
        t0 = 0
        self.viz_ee_lines(tools_pos, idx, scale, color)
        self.viz_rope_lines(keypoints, idx, scale, color='y')

        grasp_xpos = as_float(result[9])[t0]
        for name, xpos in zip(phy.o.rd.rope_grasp_eqs, grasp_xpos):
            self.viz.sphere(f'{name}_xpos', xpos, radius=hp['grasp_goal_radius'], color=(0, 1, 0, 0.4),
                            frame_id='world', idx=0)

    def recompute_candidates(self, phy: Physics):
        from time import perf_counter
        t0 = perf_counter()

        self.homotopy_locs, self.homotopy_subgoals = self.homotopy_gen.generate(phy, viz=self.viz)

        print(f'Homotopy: {self.homotopy_locs}')
        print(f'dt: {perf_counter() - t0:.3f}')
