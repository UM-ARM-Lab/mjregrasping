import networkx as nx
import numpy as np
import rerun as rr

from mjregrasping.goal_funcs import get_tool_points, ray_based_reachability, get_rope_points
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos
from mjregrasping.grasping import get_is_grasping
from mjregrasping.magnetic_fields import get_h_signature
from mjregrasping.math import softmax
from mjregrasping.mujoco_objects import parents_points
from mjregrasping.viz import Viz


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
    if i < j:
        return range(i, j)
    else:
        return range(i, j, -1)


class HomotopyGenerator(RegraspGenerator):

    def __init__(self, op_goal, skeletons, viz: Viz):
        super().__init__(op_goal, viz)
        self.skeletons = skeletons

    def generate(self, phy):
        paths = self.find_rope_robot_paths(phy)
        if paths is None:
            print("No loops found, homotopy not helpful here!")
            return np.array([-1, -1])

        # rope_points = get_rope_points(phy)
        # arm_points = [
        #     parents_points(phy.m, phy.d, "drive50"),
        #     parents_points(phy.m, phy.d, "drive10"),
        # ]
        # try:
        #     attach_pos = phy.d.body('attach').xpos
        # except KeyError:
        #     return np.array([-1, -1])
        h = []
        for path in paths:
            h_i = get_h_signature(path, self.skeletons)
            h.append(h_i)
            # rr.log_line_strip('path/0', path, ext={'hs': str(h)})

        is_grasping = get_is_grasping(phy.m)
        h = np.reshape(np.array(h) * is_grasping[:, None], -1)

        # Uniformly randomly sample a new grasp
        # reject if it's the same as the current grasp
        # reject if it's not reachable
        # check if it's H-signature is different
        allowable_is_grasping = np.array([[0, 1], [1, 0], [1, 1]])
        while True:
            candidate_is_grasping = allowable_is_grasping[self.rng.randint(0, 3)]
            candidate_locs = self.rng.uniform(0, 1, 2)
            candidate_locs = -1 * (1 - candidate_is_grasping) + candidate_locs * candidate_is_grasping

            _, _, candidate_pos = grasp_locations_to_indices_and_offsets_and_xpos(phy, candidate_locs)
            candidate_arm_points = ik(candidate_pos)

            candidate_h = []
            for candidate_arm_points_i in candidate_arm_points:
                path = self.make_closed_path(attach_pos, candidate_arm_points_i)
                h_i = get_h_signature(path, self.skeletons)
                candidate_h.append(h_i)
                # rr.log_line_strip(f'path/{i + 1}', path, ext={'hs': str(hi)})
            candidate_h = np.reshape(np.array(candidate_h) * candidate_is_grasping[:, None], -1)

            reachable_matrix = ray_based_reachability(candidate_pos, phy, tools_pos)
            candidate_not_grasping = np.logical_not(candidate_is_grasping)
            reachable_or_not_grasping = np.logical_or(np.diagonal(reachable_matrix), candidate_not_grasping)
            valid = np.all(reachable_or_not_grasping) and np.any(h != candidate_h) and np.any(candidate_is_grasping)
            if valid:
                return candidate_locs

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

        print(valid_cycles)

        # Then convert those graph cycles into closed paths in 3D
        b_point = phy.d.body('val_base').xpos
        try:
            a_point = phy.d.body('attach').xpos
        except KeyError:
            a_point = None
        arm_points = [
            parents_points(phy.m, phy.d, "drive50"),
            parents_points(phy.m, phy.d, "drive10"),
        ]
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
            rr.log_line_strip(f'robot_rope_path/{valid_cycle}', cycle_points, color=(1., 1., 1., 1.))
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
