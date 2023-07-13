from copy import copy

import numpy as np
import pysdf_tools
import rerun as rr
from bayes_opt import BayesianOptimization

from arc_utilities.path_utils import densify
from mjregrasping.goal_funcs import get_rope_points, check_should_be_open
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.grasping import get_grasp_locs, get_is_grasping, activate_grasp
from mjregrasping.ik import get_reachability_cost, HARD_CONSTRAINT_PENALTY, eq_sim_ik
from mjregrasping.params import hp
from mjregrasping.homotopy_checker import FirstOrderComparer
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
                candidate_locs, candidate_subgoals = args_to_locs_and_subgoals(kwargs)
                # # DEBUGGING:
                # candidate_locs = np.array([0.5])
                # candidate_subgoals = np.array([[1.1, 0.9, 0.30]])

                # kwargs keys are based on the bounds dict
                candidate_idx, _, candidate_pos = grasp_locations_to_indices_and_offsets_and_xpos(phy, candidate_locs)

                for tool_name, xpos, subgoal in zip(phy.o.rd.tool_sites, candidate_pos, candidate_subgoals):
                    self.viz.sphere(f'first_order/{tool_name}_xpos', xpos, hp['grasp_goal_radius'], 'world', 'w', 0)
                    self.viz.sphere(f'first_order/{tool_name}_subgoal', subgoal, hp['grasp_goal_radius'], 'world', 'c',
                                    0)

                # Construct phy_ik to match the candidate grasp
                phy_regrasp, reached_regrasp = self.sim_regrasp(phy, candidate_locs, candidate_is_grasping,
                                                                candidate_pos)
                # FIXME: doesn't generalize to multiple grippers
                phy_subgoal, reached_subgoal = self.sim_subgoals(phy_regrasp, candidate_locs, candidate_is_grasping,
                                                                 candidate_subgoals)

                path1_dense = np.array(densify(path1, 4 * self.sdf.GetResolution()))
                path2 = copy(get_rope_points(phy_subgoal))
                path2_dense = np.array(densify(path2, 4 * self.sdf.GetResolution()))

                reachability_cost = sum([
                    get_reachability_cost(phy, phy_regrasp, reached_regrasp, candidate_locs, candidate_is_grasping),
                    get_reachability_cost(phy_regrasp, phy_subgoal, reached_subgoal, candidate_locs,
                                          candidate_is_grasping)
                ])

                path1_in_collision = any([self.sdf.GetValueByCoordinates(*p)[0] < -self.sdf.GetResolution() for p in path1_dense])
                if path1_in_collision:
                    raise ValueError("Path 1 is in collision, this should never happen!")

                self.viz.lines(path1_dense, 'first_order/path1', 0, 0.005, 'r')
                self.viz.lines(path2_dense, 'first_order/path2', 0, 0.005, 'orange')
                # indices_path = get_first_order_homotopy_points(self.sdf, path1_dense, path2_dense)
                # if len(indices_path) > 0:
                #     for i1, i2 in indices_path:
                #         p1 = path1_dense[i1]
                #         p2 = path2_dense[i2]
                #         self.viz.lines([p1, p2], 'first_order/indices_path', 0, 0.005, 'white')

                homotopy_cost = HARD_CONSTRAINT_PENALTY if comparer.check_equal(path1_dense, path2_dense) else 0

                cost = reachability_cost + homotopy_cost

                rr.log_scalar("first_order_costs/reachability", reachability_cost)
                rr.log_scalar("first_order_costs/homotopy", homotopy_cost)
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
            print("All candidates were first-order-homologous")
            return None, None

        return best_locs, best_subgoals

    def sim_regrasp(self, phy: Physics, candidate_locs, candidate_is_grasping, candidate_pos):
        phy_ik = phy.copy_all()

        # First deactivate any grippers that need to change location or are simply not
        # supposed to be grasping given candidate_locs
        desired_open = check_should_be_open(current_grasp_locs=get_grasp_locs(phy),
                                            current_is_grasping=get_is_grasping(phy),
                                            desired_locs=candidate_locs,
                                            desired_is_grasping=candidate_is_grasping)
        for eq_name, desired_open_i in zip(phy_ik.o.rd.rope_grasp_eqs, desired_open):
            eq = phy_ik.m.eq(eq_name)
            if desired_open_i:
                eq.active = 0

        # Disable collisions for the rope, except for the floor
        for rope_geom_name in phy_ik.o.rope.geom_names:
            rope_geom = phy_ik.m.geom(rope_geom_name)
            rope_geom.contype = 2
            rope_geom.conaffinity = 2
        floor_geom = phy_ik.m.geom('floor')
        floor_geom.contype = 2
        floor_geom.conaffinity = 2

        # Estimate reachability using eq constraints in mujoco
        reached = eq_sim_ik(candidate_is_grasping, candidate_pos, phy_ik, viz=self.viz)

        return phy_ik, reached

    def sim_subgoals(self, phy: Physics, candidate_locs, candidate_is_grasping, candidate_pos):
        phy_ik = phy.copy_all()

        # Activate any grippers that need to be grasping
        for eq_name, is_grasping_i, loc_i in zip(phy_ik.o.rd.rope_grasp_eqs, candidate_is_grasping, candidate_locs):
            eq = phy_ik.m.eq(eq_name)
            if is_grasping_i:
                activate_grasp(phy_ik, eq_name, loc_i)

        # Disable collisions for the rope, except for the floor
        for rope_geom_name in phy_ik.o.rope.geom_names:
            rope_geom = phy_ik.m.geom(rope_geom_name)
            rope_geom.contype = 2
            rope_geom.conaffinity = 2
        floor_geom = phy_ik.m.geom('floor')
        floor_geom.contype = 2
        floor_geom.conaffinity = 2

        reached = eq_sim_ik(candidate_is_grasping, candidate_pos, phy_ik, viz=self.viz)

        return phy_ik, reached


