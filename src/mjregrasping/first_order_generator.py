from copy import copy

import numpy as np
import pysdf_tools
import rerun as rr
from pymjregrasping_cpp import get_first_order_homotopy_points
from bayes_opt import BayesianOptimization

from arc_utilities.path_utils import path_length, densify
from mjregrasping.goal_funcs import get_rope_points, get_contact_cost
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos, sln_to_locs
from mjregrasping.ik import create_eq_and_sim_ik, get_reachability_cost, HARD_CONSTRAINT_PENALTY
from mjregrasping.path_comparer import FirstOrderComparer
from mjregrasping.physics import Physics
from mjregrasping.regrasp_generator import RegraspGenerator, get_allowable_is_grasping
from mjregrasping.viz import Viz


class FirstOrderGenerator(RegraspGenerator):

    def __init__(self, op_goal, sdf: pysdf_tools.SignedDistanceField, viz: Viz):
        super().__init__(op_goal, viz)
        self.sdf = sdf

    def generate(self, phy: Physics):
        """
        Tries to find a new grasp that changes the homotopy. The homotopy is defined by first order homotopy.
        """
        comparer = FirstOrderComparer(self.sdf)
        path1 = copy(get_rope_points(phy))

        # NOTE: what if there is no way to change homotopy? We need some kind of stopping condition.
        #  Can we prove that there is always a way to change homotopy? Or prove a correct stopping condition?
        best_locs = None
        best_cost = np.inf
        best_subgoals = None
        for candidate_is_grasping in get_allowable_is_grasping(phy.o.rd.n_g):
            bounds = {}
            for tool_name, is_grasping_i in zip(phy.o.rd.tool_sites, candidate_is_grasping):
                if is_grasping_i:
                    bounds[tool_name] = (0, 1)
                    bounds[tool_name + '_subgoal_x'] = (0, 2)  # TODO: proper workspace bounds
                    bounds[tool_name + '_subgoal_y'] = (-2, 2)  # TODO: proper workspace bounds
                    bounds[tool_name + '_subgoal_z'] = (0, 0.1)  # TODO: proper workspace bounds

            def _cost(**kwargs):
                candidate_locs = []
                candidate_subgoals = []
                for tool_name, is_grasping_i in zip(phy.o.rd.tool_sites, candidate_is_grasping):
                    if is_grasping_i:
                        candidate_locs.append(kwargs[tool_name])
                        candidate_subgoals.append(np.array([kwargs[tool_name + '_subgoal_x'],
                                                            kwargs[tool_name + '_subgoal_y'],
                                                            kwargs[tool_name + '_subgoal_z']]))
                    else:
                        candidate_locs.append(-1)
                        candidate_subgoals.append(None)
                candidate_locs = np.array(candidate_locs)
                candidate_subgoals = np.array(candidate_subgoals)

                # kwargs keys are based on the bounds dict
                _, _, candidate_pos = grasp_locations_to_indices_and_offsets_and_xpos(phy, candidate_locs)

                # Construct phy_ik to match the candidate grasp
                phy_regrasp, reached_regrasp = create_eq_and_sim_ik(phy, candidate_is_grasping, candidate_pos,
                                                                    viz=None)
                # self.viz.viz(phy_regrasp, is_planning=True)
                # FIXME: doesn't generalize to multiple grippers
                phy_subgoal, reached_subgoal = create_eq_and_sim_ik(phy_regrasp, candidate_is_grasping,
                                                                    candidate_subgoals, viz=None)
                # self.viz.viz(phy_subgoal, is_planning=True)

                # The real challenge is what "next" rope state (b1 in Dale's terminology) to use.
                # The ideal thing would be to run the MPC controller to reach the subgoal, but that would be too slow
                # Straight line is probably a terrible approximation.
                path2 = [
                    path1[0],
                    candidate_subgoals[0],  # FIXME: doesn't generalize to multiple grippers
                    path1[-1]
                ]

                path1_dense = np.array(densify(path1, 4 * self.sdf.GetResolution()))
                path2_dense = np.array(densify(path2, 4 * self.sdf.GetResolution()))

                reachability_cost = sum([
                    get_reachability_cost(phy, phy_regrasp, reached_regrasp, candidate_locs, candidate_is_grasping),
                    get_reachability_cost(phy_regrasp, phy_subgoal, reached_subgoal, candidate_locs,
                                          candidate_is_grasping)
                ])

                path1_in_collision = any([self.sdf.GetValueByCoordinates(*p)[0] < 0 for p in path1_dense])
                if path1_in_collision:
                    raise ValueError("Path 1 is in collision, this should never happen!")
                path2_in_collision = any([self.sdf.GetValueByCoordinates(*p)[0] < 0 for p in path2_dense])

                self.viz.lines(path1_dense, 'first_order/path1', 0, 0.005, 'r')
                self.viz.lines(path2_dense, 'first_order/path2', 0, 0.005, 'orange')
                # indices_path = get_first_order_homotopy_points(self.sdf, path1_dense, path2_dense)
                # if len(indices_path) > 0:
                #     for i1, i2 in indices_path:
                #         p1 = path1_dense[i1]
                #         p2 = path2_dense[i2]
                #         self.viz.lines([p1, p2], 'first_order/indices_path', 0, 0.005, 'white')

                homotopy_cost = HARD_CONSTRAINT_PENALTY if comparer.check_equal(path1_dense, path2_dense) else 0
                path2_collision_cost = HARD_CONSTRAINT_PENALTY if path2_in_collision else 0
                stretch_cost = max((path_length(path2) - path_length(path1)) ** 2, 0)  # don't penalize if path2 is shorter
                contact_cost = get_contact_cost(phy_regrasp) + get_contact_cost(phy_subgoal)

                cost = reachability_cost + homotopy_cost + path2_collision_cost + contact_cost + stretch_cost

                rr.log_scalar("first_order_costs/reachability", reachability_cost)
                rr.log_scalar("first_order_costs/contact", contact_cost)
                rr.log_scalar("first_order_costs/homotopy", homotopy_cost)
                rr.log_scalar("first_order_costs/path2_collision", path2_collision_cost)
                rr.log_scalar("first_order_costs/stretch", stretch_cost)
                rr.log_scalar("first_order_costs/total_cost", cost)
                return -cost

            opt = BayesianOptimization(f=_cost, pbounds=bounds, verbose=0, random_state=self.rng.randint(0, 1000),
                                       allow_duplicate_points=True)
            opt.maximize(n_iter=15, init_points=5)
            sln = opt.max
            cost = -sln['target']  # BayesOpt uses maximization, so we need to negate to get cost
            locs = sln_to_locs(sln['params'], candidate_is_grasping)
            # subgoals = sln_to_subgoals(sln['params'], phy.o.rd.tool_sites, candidate_is_grasping)
            subgoals = sln_to_subgoals(sln['params'], candidate_is_grasping, phy.o.rd.tool_sites)

            # _, _, candidate_pos = grasp_locations_to_indices_and_offsets_and_xpos(phy, locs)
            # phy_ik, _ = create_eq_and_sim_ik(phy, self.tool_names, self.gripper_to_world_eq_names,
            #                                  candidate_is_grasping, candidate_pos)
            # self.viz.viz(phy_ik, is_planning=True)

            if cost < best_cost:
                best_locs = locs
                best_cost = cost
                best_subgoals = subgoals

        if best_cost >= HARD_CONSTRAINT_PENALTY:
            print("All candidates were homologous")
            return None

        return best_locs, best_subgoals


def sln_to_subgoals(params, candidate_is_grasping, tool_names):
    subgoals = []
    for tool_name, is_grasping_i in zip(tool_names, candidate_is_grasping):
        if is_grasping_i:
            subgoals.append(np.array([params[tool_name + '_subgoal_x'],
                                      params[tool_name + '_subgoal_y'],
                                      params[tool_name + '_subgoal_z']]))
        else:
            subgoals.append(None)
    return subgoals
