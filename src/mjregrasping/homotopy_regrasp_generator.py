from copy import copy
from typing import Dict

import numpy as np
from bayes_opt import BayesianOptimization

from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos, make_full_locs, sln_to_locs
from mjregrasping.ik import HARD_CONSTRAINT_PENALTY, get_reachability_cost, eq_sim_ik
from mjregrasping.path_comparer import TrueHomotopyComparer
from mjregrasping.physics import Physics
from mjregrasping.regrasp_generator import RegraspGenerator, get_allowable_is_grasping
from mjregrasping.rerun_visualizer import log_skeletons
from mjregrasping.viz import Viz


class HomotopyGenerator(RegraspGenerator):

    def __init__(self, op_goal, skeletons: Dict, viz: Viz):
        super().__init__(op_goal, viz)
        self.skeletons = skeletons

    def generate(self, phy: Physics):
        """
        Tries to find a new grasp that changes the homotopy. The homotopy is defined by both the skeleton of the scene
        and the "loops" created by the robots arms and the rope.
        """
        comparer = TrueHomotopyComparer(self.skeletons)
        # NOTE: not sure this path comparer is event a good abstraction
        initial_rope_points = copy(get_rope_points(phy))
        h = comparer.get_signature(phy, initial_rope_points)
        log_skeletons(self.skeletons, stroke_width=0.01, color=[0, 1.0, 0, 1.0])

        # NOTE: what if there is no way to change homotopy? We need some kind of stopping condition.
        #  Can we prove that there is always a way to change homotopy? Or prove a correct stopping condition?
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
                _, _, candidate_pos = grasp_locations_to_indices_and_offsets_and_xpos(phy, candidate_locs)

                phy_ik = phy.copy_all()
                # deactivate any eqs where the gripper should release
                for eq_name, is_grasping_i in zip(phy_ik.o.rd.rope_grasp_eqs, candidate_is_grasping):
                    if not is_grasping_i:
                        phy_ik.m.eq(eq_name).active = False
                reached = eq_sim_ik(candidate_is_grasping, candidate_pos, phy_ik, viz=None)
                # now activate the eqs for grippers that should be grasping
                for eq_name, is_grasping_i in zip(phy_ik.o.rd.rope_grasp_eqs, candidate_is_grasping):
                    if is_grasping_i:
                        phy_ik.m.eq(eq_name).active = True
                # self.viz.viz(phy_ik, is_planning=True)

                reachability_cost = get_reachability_cost(phy, phy_ik, reached, candidate_locs, candidate_is_grasping)

                candidate_h = comparer.get_signature(phy_ik, initial_rope_points, log_loops=True)
                homotopy_cost = HARD_CONSTRAINT_PENALTY if comparer.check_equal(h, candidate_h) else 0

                cost = reachability_cost + homotopy_cost
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
            print("All candidates were homologous")
            return None, None

        _, _, pos = grasp_locations_to_indices_and_offsets_and_xpos(phy, best_locs)
        pull_dir = self.op_goal.goal_point - pos
        pull_dir /= np.linalg.norm(pull_dir)
        subgoals = pos + pull_dir * 1.0
        # TODO: make the subgoals consistent with "is grasping"

        return best_locs, subgoals
