import numpy as np

from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.grasping import get_is_grasping
from mjregrasping.ik import eq_sim_ik
from mjregrasping.math import softmax
from mjregrasping.regrasp_generator import RegraspGenerator
from mjregrasping.viz import Viz


class SlackGenerator(RegraspGenerator):

    def __init__(self, op_goal, viz: Viz):
        super().__init__(op_goal, viz)

    def generate(self, phy):
        is_grasping = get_is_grasping(phy.m)
        not_grasping = 1 - is_grasping + 1e-3
        candidate_gripper_p = softmax(not_grasping, 1.0, sub_max=True)
        candidate_is_grasping = self.rng.binomial(1, candidate_gripper_p)
        candidates_locs = np.linspace(0, 1, 10)
        candidates_bodies, candidates_offsets, candidates_xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy,
                                                                                                                 candidates_locs)
        phy_ik = phy.copy_all()
        gripper_to_world_eq_names = ['left_world', 'right_world']
        tool_names = ['left_tool', 'right_tool']
        reachability = np.zeros([len(tool_names), len(candidates_xpos)], dtype=bool)
        for i, tool_name in enumerate(tool_names):
            gripper_to_world_eq = phy_ik.m.eq(gripper_to_world_eq_names[i])
            gripper_to_world_eq.active = 1
            for j, candidate_pos in enumerate(candidates_xpos):
                gripper_to_world_eq.data[3:6] = candidate_pos
                reached = eq_sim_ik([tool_name], [candidate_is_grasping[i]], [candidate_pos], phy_ik, viz=None)
                q_before = phy.d.qpos[phy.o.val.qpos_indices]
                q_after = phy_ik.d.qpos[phy_ik.o.val.qpos_indices]
                reachability_cost = np.linalg.norm(q_after - q_before)
                reachability[i, j] = reachability_cost if reached else 1000
            gripper_to_world_eq.active = 0
        # tools_pos = get_tool_points(phy)
        # is_reachable = ray_based_reachability(candidates_xpos, phy, tools_pos)

        geodesics_costs = np.square(candidates_locs - self.op_goal.loc)
        combined_costs = geodesics_costs + reachability_cost
        best_idx = np.argmin(combined_costs, axis=-1)
        best_candidate_locs = candidates_locs[best_idx]
        best_candidate_locs = best_candidate_locs * candidate_is_grasping + -1 * (1 - candidate_is_grasping)
        return best_candidate_locs
