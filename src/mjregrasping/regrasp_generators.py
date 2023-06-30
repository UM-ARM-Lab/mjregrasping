import time

import mujoco
import networkx as nx
import numpy as np
import rerun as rr
from numpy.linalg import norm
from pybio_ik import BioIK

from mjregrasping.goal_funcs import get_tool_points, ray_based_reachability, get_rope_points
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.grasping import get_is_grasping, get_grasp_eqs
from mjregrasping.magnetic_fields import get_h_signature
from mjregrasping.math import softmax
from mjregrasping.mujoco_objects import parents_points
from mjregrasping.params import hp
from mjregrasping.viz import Viz

BOTH_ARMS_IK_Q_INDICES = np.array([0, 1,
                                   2, 3, 4, 5, 6, 7, 8,
                                   11, 12, 13, 14, 15, 16, 17,
                                   ])
IK_OFFSET = np.array([0, 0, 0.145])


class RegraspGenerator:

    def __init__(self, op_goal, viz: Viz):
        self.op_goal = op_goal
        self.viz = viz
        self.rng = np.random.RandomState(0)

    def generate(self, phy):
        """
        Args:
            phy: Current state of the world

        Returns:
            the next grasp locations. This is a vector of size [n_g] where the element is -1 if there is no grasp,
            and between 0 and 1 if there is a grasp. A grasp of 0 means one end, 0.5 means the middle, and 1 means
            the other end.
        """
        raise NotImplementedError()


def pairwise(x):
    return zip(x[:-1], x[1:])


def from_to(i, j):
    """ Inclusive of both i and j """
    if i < j:
        return range(i, j + 1)
    else:
        return range(i, j - 1, -1)


class HomotopyGenerator(RegraspGenerator):

    def __init__(self, op_goal, skeletons, viz: Viz):
        super().__init__(op_goal, viz)
        self.skeletons = skeletons
        self.tool_names = ["left_tool", "right_tool"]
        self.tool_bodies = ["drive50", "drive10"]
        self.gripper_to_world_eq_names = ['left_world', 'right_world']
        self.bio_ik = BioIK("robot_description")

    def generate(self, phy):
        paths = self.find_rope_robot_paths(phy)
        if paths is None:
            print("No loops found, homotopy not helpful here!")
            return np.array([-1, -1])
        for p in paths:
            rr.log_line_strip('initial_paths', p)

        h = self.get_h_signature_for_paths(paths)

        allowable_is_grasping = np.array([
            [0, 1],
            [1, 0],
            # [0, 0],
            # [1, 1],
        ])
        # NOTE: what if there is no way to change homotopy? We need some kind of stopping condition.
        #  Can we prove that there is always a way to change homotopy? Or prove a correct stopping condition?
        while True:
            candidate_is_grasping = allowable_is_grasping[self.rng.randint(0, len(allowable_is_grasping))]
            candidate_locs = self.rng.uniform(0, 1, 2)
            candidate_locs = -1 * (1 - candidate_is_grasping) + candidate_locs * candidate_is_grasping
            candidate_indices, _, candidate_pos = grasp_locations_to_indices_and_offsets_and_xpos(phy, candidate_locs)

            # Construct phy_ik to match the candidate grasp
            phy_ik = phy.copy_data()
            grasp_eqs = get_grasp_eqs(phy_ik.m)
            ik_targets = {}
            for i in range(hp['n_g']):
                tool_name = self.tool_names[i]
                gripper_to_world_eq_name = self.gripper_to_world_eq_names[i]
                eq = grasp_eqs[i]
                # Set eq_active and qpos depending on the grasp
                if candidate_is_grasping[i]:
                    eq.active = 1
                    eq.obj2id = candidate_indices[i]
                    ik_targets[tool_name] = candidate_pos[i] - IK_OFFSET
                    gripper_to_world_eq = phy_ik.m.eq(gripper_to_world_eq_name)
                    gripper_to_world_eq.active = 1
                    gripper_to_world_eq.data[3:6] = candidate_pos[i]
                else:
                    eq.active = 0

            # Estimate reachability using eq constraints in mujoco
            for _ in range(10):
                mujoco.mj_step(phy_ik.m, phy_ik.d, nstep=25)
                # Check if the grasping grippers are near their targets
                reached = True
                for i in range(hp['n_g']):
                    if candidate_is_grasping[i]:
                        d = norm(phy_ik.d.site(self.tool_names[i]).xpos - candidate_pos[i])
                        if d > 0.01:
                            reached = False
                # self.viz.viz(phy_ik, is_planning=True)
                if reached:
                    break
            if not reached:
                continue

            candidate_paths = self.find_rope_robot_paths(phy_ik)
            if paths is None:  # shouldn't happen, having no grasps isn't allowed, but doesn't hurt to check.
                continue

            for name, xpos in zip(self.tool_names, candidate_pos):
                rr.log_point(f'tool_pos/{name}', xpos, radius=0.04)
            for p in candidate_paths:
                rr.log_line_strip('candidate_path', p)

            candidate_h = self.get_h_signature_for_paths(candidate_paths)

            valid = np.any(h != candidate_h) and np.any(candidate_is_grasping)
            if valid:
                self.viz.viz(phy_ik, is_planning=True)
                return candidate_locs

    def get_h_signature_for_paths(self, paths):
        return np.array([get_h_signature(path, self.skeletons) for path in paths])

    def find_rope_robot_paths(self, phy):
        # First construct a graph based on the current constraints/grasps
        is_grasping = get_is_grasping(phy.m)
        graph = nx.Graph()
        nodes = ['b']
        for i, is_grasping_i in enumerate(is_grasping):
            if is_grasping_i:
                nodes.append(f'g{i}')
        try:
            phy.m.eq("attach")  # check if there is NO attach constraint by exception
            nodes.append('a')
        except KeyError:
            pass
        graph.add_nodes_from(nodes)
        for i in graph.nodes:
            for j in graph.nodes:
                if i != j:
                    graph.add_edge(i, j)

        valid_cycles = []
        for cycle in nx.simple_cycles(graph, length_bound=3):
            # Cycle must be length 3 and contain the robot base.
            # There could be a cycle like ['g0', 'g1', 'a'], but we don't need to consider that
            # Since that only involves the points on the rope, so it will always be in the same homotopy class
            if len(cycle) == 3 and 'b' in cycle:
                valid_cycles.append(cycle)

        if len(valid_cycles) == 0:
            return None

        # print(valid_cycles)

        # Then convert those graph cycles into closed paths in 3D
        b_point = phy.d.body('val_base').xpos
        try:
            a_point = phy.d.body('attach').xpos
        except KeyError:
            a_point = None
        arm_points = [parents_points(phy.m, phy.d, tool_body) for tool_body in self.tool_bodies]
        grasped_indices = np.array([
            int(phy.m.eq("left").obj2id),
            int(phy.m.eq("right").obj2id)
        ])
        rope_start_body_index = np.min(phy.o.rope.body_indices)
        rope_points_indices = grasped_indices - rope_start_body_index
        rope_points_indices = is_grasping * rope_points_indices + (1 - is_grasping) * -1
        attach_index = int(phy.m.eq("attach").obj2id) - rope_start_body_index
        rope_points = get_rope_points(phy)

        all_points = []
        for valid_cycle in valid_cycles:
            valid_cycle = valid_cycle + [valid_cycle[0]]
            # We now need to convert this into a path in 3D, which we can use to compute the H-signature
            # The points are based on the "edges" in the cycle.
            # An edge like "b->g0" or "g1->b" means the points come from FK or IK
            # An edge like "g0->a" means the points come the rope. Specifically, we use the positions of the rope bodies
            # which lie between the body index of g0 and the body index of a, which we get from the EQ constraints.
            # the order of the bodies depends on the order of the edge, i.e "g0->a" vs "a->g0".
            # Finally, an edge like "b->a" is simply the robot base position and the attach position, although oreder
            # does still matter.
            cycle_points = []
            for edge in pairwise(valid_cycle):
                if edge == ('b', 'a'):
                    cycle_points.append(b_point)
                    cycle_points.append(a_point)
                elif edge == ('a', 'b'):
                    cycle_points.append(a_point)
                    cycle_points.append(b_point)
                elif edge == ('b', 'g0'):
                    cycle_points.extend(arm_points[0][::-1])
                elif edge == ('g0', 'b'):
                    cycle_points.extend(arm_points[0])
                elif edge == ('b', 'g1'):
                    cycle_points.extend(arm_points[1][::-1])
                elif edge == ('g1', 'b'):
                    cycle_points.extend(arm_points[1])
                elif edge == ('g0', 'a'):
                    cycle_points.extend(rope_points[from_to(rope_points_indices[0], attach_index)])
                elif edge == ('a', 'g0'):
                    cycle_points.extend(rope_points[from_to(attach_index, rope_points_indices[0])])
                elif edge == ('g1', 'a'):
                    cycle_points.extend(rope_points[from_to(rope_points_indices[1], attach_index)])
                elif edge == ('a', 'g1'):
                    cycle_points.extend(rope_points[from_to(attach_index, rope_points_indices[1])])
                elif edge == ('g0', 'g1'):
                    cycle_points.extend(rope_points[from_to(rope_points_indices[0], rope_points_indices[1])])
                elif edge == ('g1', 'g0'):
                    cycle_points.extend(rope_points[from_to(rope_points_indices[1], rope_points_indices[0])])
                else:
                    raise RuntimeError(f"Unknown edge {edge}")
            cycle_points = np.stack(cycle_points, axis=0)
            all_points.append(cycle_points)
        return all_points


class SlackGenerator(RegraspGenerator):

    def __init__(self, op_goal, viz: Viz):
        super().__init__(op_goal, viz)

    def generate(self, phy):
        tools_pos = get_tool_points(phy)
        is_grasping = get_is_grasping(phy.m)
        not_grasping = 1 - is_grasping
        candidate_gripper_p = softmax(not_grasping, 0.1, sub_max=False)
        candidate_is_grasping = self.rng.binomial(1, candidate_gripper_p)
        candidates_locs = np.linspace(0, 1, 10)
        candidates_bodies, candidates_offsets, candidates_xpos = grasp_locations_to_indices_and_offsets_and_xpos(phy,
                                                                                                                 candidates_locs)
        is_reachable = ray_based_reachability(candidates_xpos, phy, tools_pos)
        geodesics_costs = np.square(candidates_locs - self.op_goal.loc)
        combined_costs = geodesics_costs + 1000 * (1 - is_reachable)
        best_idx = np.argmin(combined_costs, axis=-1)
        best_candidate_locs = candidates_locs[best_idx]
        best_candidate_locs = best_candidate_locs * candidate_is_grasping + -1 * (1 - candidate_is_grasping)
        return best_candidate_locs
