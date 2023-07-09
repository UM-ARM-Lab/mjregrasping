import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.target_space import NotUniqueError

from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos, make_full_locs, sln_to_locs
from mjregrasping.ik import get_reachability_cost, create_eq_and_sim_ik, HARD_CONSTRAINT_PENALTY
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.regrasp_generator import RegraspGenerator, get_allowable_is_grasping
from mjregrasping.viz import Viz


class SlackGenerator(RegraspGenerator):

    def __init__(self, op_goal, viz: Viz):
        super().__init__(op_goal, viz)

    def generate(self, phy: Physics):
        best_locs = None
        best_cost = np.inf
        for candidate_is_grasping in get_allowable_is_grasping(phy.o.rd.n_g):
            bounds = {}
            for tool_name, is_grasping_i in zip(phy.o.rd.tool_sites, candidate_is_grasping):
                if is_grasping_i:
                    bounds[tool_name] = (0, 1)

            def _cost(**locs):
                locs_where_grasping = list(locs.values())  # Only contains locs for the grasping grippers
                candidate_locs = make_full_locs(locs_where_grasping, candidate_is_grasping)
                candidate_idx, _, candidate_pos = grasp_locations_to_indices_and_offsets_and_xpos(phy, candidate_locs)

                # Construct phy_ik to match the candidate grasp
                phy_ik, reached = create_eq_and_sim_ik(phy, candidate_is_grasping, candidate_idx, candidate_pos,
                                                       viz=None)
                # self.viz.viz(phy_ik, is_planning=True)

                reachability_cost = get_reachability_cost(phy, phy_ik, reached, candidate_locs, candidate_is_grasping)

                geodesics_cost = np.min(np.square(candidate_locs - self.op_goal.loc))
                cost = reachability_cost + geodesics_cost * hp['geodesic_weight']
                # BayesOpt uses maximization, so we need to negate the cost
                return -cost

            opt = BayesianOptimization(f=_cost, pbounds=bounds, verbose=0, random_state=self.rng.randint(0, 1000))
            try:
                opt.maximize(n_iter=8, init_points=5)
            except NotUniqueError:
                pass
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
