import numpy as np
from bayes_opt import BayesianOptimization

from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos, make_full_locs, sln_to_locs
from mjregrasping.ik import get_reachability_cost, create_eq_and_sim_ik, HARD_CONSTRAINT_PENALTY
from mjregrasping.params import ALLOWABLE_IS_GRASPING, hp
from mjregrasping.regrasp_generator import RegraspGenerator
from mjregrasping.viz import Viz


class SlackGenerator(RegraspGenerator):

    def __init__(self, op_goal, viz: Viz):
        super().__init__(op_goal, viz)
        self.tool_names = ['left_tool', 'right_tool']
        self.tool_bodies = ["drive50", "drive10"]
        self.gripper_to_world_eq_names = ['left_world', 'right_world']

    def generate(self, phy):
        best_locs = None
        best_cost = np.inf
        for candidate_is_grasping in ALLOWABLE_IS_GRASPING:
            bounds = {}
            for tool_name, is_grasping_i in zip(self.tool_names, candidate_is_grasping):
                if is_grasping_i:
                    bounds[tool_name] = (0, 1)

            def _cost(**locs):
                locs_where_grasping = list(locs.values())  # Only contains locs for the grasping grippers
                candidate_locs = make_full_locs(locs_where_grasping, candidate_is_grasping)
                candidate_idx, _, candidate_pos = grasp_locations_to_indices_and_offsets_and_xpos(phy, candidate_locs)

                # Construct phy_ik to match the candidate grasp
                phy_ik, reached = create_eq_and_sim_ik(phy, self.tool_names, self.gripper_to_world_eq_names,
                                                       candidate_is_grasping, candidate_idx, candidate_pos,
                                                       viz=None)
                # self.viz.viz(phy_ik, is_planning=True)

                reachability_cost = get_reachability_cost(phy, phy_ik, reached, candidate_locs, candidate_is_grasping)

                geodesics_cost = np.min(np.square(candidate_locs - self.op_goal.loc))
                cost = reachability_cost + geodesics_cost * hp['geodesic_weight']
                # BayesOpt uses maximization, so we need to negate the cost
                return -cost

            opt = BayesianOptimization(f=_cost, pbounds=bounds, verbose=0, random_state=self.rng.randint(0, 1000),
                                       allow_duplicate_points=True)
            opt.maximize(n_iter=8, init_points=5)
            sln = opt.max
            cost = -sln['target']  # BayesOpt uses maximization, so we need to negate to get cost
            locs = sln_to_locs(sln['params'], candidate_is_grasping)

            if cost < best_cost:
                best_locs = locs
                best_cost = cost

        if best_cost >= HARD_CONSTRAINT_PENALTY:
            print("No candidates were reachable")
            return None
        return best_locs

        ################
        # is_grasping = get_is_grasping(phy.m)
        # # TODO: instead of randomly sampling with a bias away from current grasp, do exhaustive search in binary space?
        # not_grasping = 1 - is_grasping + 1e-3
        # candidate_gripper_p = softmax(not_grasping, 1.0, sub_max=True)
        # candidate_is_grasping = self.rng.binomial(1, candidate_gripper_p)
        # candidates_locs = np.linspace(0, 1, 10)
        # candidates_bodies, candidates_offsets, candidates_xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy,
        #                                                                                                          candidates_locs)
        # phy_ik = phy.copy_all()
        # gripper_to_world_eq_names = ['left_world', 'right_world']
        # reachability_costs = np.zeros([len(tool_names), len(candidates_xpos)], dtype=bool)
        # for i, tool_name in enumerate(tool_names):
        #     gripper_to_world_eq = phy_ik.m.eq(gripper_to_world_eq_names[i])
        #     gripper_to_world_eq.active = 1
        #     for j, candidate_pos in enumerate(candidates_xpos):
        #         gripper_to_world_eq.data[3:6] = candidate_pos
        #         reached = eq_sim_ik([tool_name], [candidate_is_grasping[i]], [candidate_pos], phy_ik, viz=None)
        #         reachability_costs[i, j] = get_reachability_cost(phy, phy_ik, reached, candidates_locs_i, candidate_is_grasping)
        #     gripper_to_world_eq.active = 0
        #
        # geodesics_costs = np.square(candidates_locs - self.op_goal.loc)
        # combined_costs = geodesics_costs + reachability_costs
        # best_idx = np.argmin(combined_costs, axis=-1)
        # best_candidate_locs = candidates_locs[best_idx]
        # best_candidate_locs = best_candidate_locs * candidate_is_grasping + -1 * (1 - candidate_is_grasping)
        # if np.all(best_candidate_locs == -1):
        #     return None
        # return best_candidate_locs
