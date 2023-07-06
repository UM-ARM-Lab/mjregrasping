from copy import copy
from bayes_opt import BayesianOptimization, Events
import cma
from typing import List

import networkx as nx
import numpy as np
import rerun as rr
from pybio_ik import BioIK

from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.grasp_conversions import grasp_locations_to_indices_and_offsets_and_xpos, grasp_indices_to_locations
from mjregrasping.grasping import get_grasp_eqs, get_rope_length
from mjregrasping.ik import eq_sim_ik
from mjregrasping.magnetic_fields import get_h_signature
from mjregrasping.mujoco_objects import parents_points
from mjregrasping.params import hp
from mjregrasping.regrasp_generator import RegraspGenerator
from mjregrasping.viz import Viz

HARD_CONSTRAINT_PENALTY = 1e3

IK_OFFSET = np.array([0, 0, 0.145])


def pairwise(x):
    return zip(x[:-1], x[1:])


def from_to(i, j):
    """ Inclusive of both i and j """
    if i < j:
        return range(i + 1, j + 1)
    else:
        return range(i, j, -1)


def h_equal(h1: List, h2: List):
    if np.shape(h1) != np.shape(h2):
        return False
    try:
        return np.all(h1 == h2)
    except ValueError:
        return np.allclose(np.sort(h1, axis=0), np.sort(h2, axis=0))


class HomotopyGenerator(RegraspGenerator):

    def __init__(self, op_goal, skeletons, viz: Viz):
        super().__init__(op_goal, viz)
        self.skeletons = skeletons
        self.tool_names = ["left_tool", "right_tool"]
        self.tool_bodies = ["drive50", "drive10"]
        self.gripper_to_world_eq_names = ['left_world', 'right_world']
        self.bio_ik = BioIK("robot_description")

    def generate(self, phy):
        initial_rope_points = copy(get_rope_points(phy))
        _, h = self.get_h_signature(phy, initial_rope_points)

        allowable_is_grasping = np.array([
            [0, 1],
            [1, 0],
            [1, 1],
            # [0, 0],
        ])

        # NOTE: what if there is no way to change homotopy? We need some kind of stopping condition.
        #  Can we prove that there is always a way to change homotopy? Or prove a correct stopping condition?
        best_locs = None
        best_cost = np.inf
        for candidate_is_grasping in allowable_is_grasping:
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
                                                       candidate_is_grasping, candidate_idx, candidate_pos)
                q_before = phy.d.qpos[phy.o.val.qpos_indices]
                q_after = phy_ik.d.qpos[phy_ik.o.val.qpos_indices]
                reachability_cost = np.linalg.norm(q_after - q_before) if reached else HARD_CONSTRAINT_PENALTY

                _, candidate_h = self.get_h_signature(phy_ik, initial_rope_points)
                homotopy_cost = HARD_CONSTRAINT_PENALTY if h_equal(h, candidate_h) else 0

                cost = reachability_cost + homotopy_cost
                # BayesOpt uses maximization, so we need to negate the cost
                return -cost

            opt = BayesianOptimization(f=_cost, pbounds=bounds, verbose=2, random_state=1)
            opt.maximize(n_iter=10, init_points=3)
            sln = opt.max
            cost = -sln['target']  # BayesOpt uses maximization, so we need to negate to get cost
            locs = sln_to_locs(sln['params'], candidate_is_grasping)

            # candidate_idx, _, candidate_pos = grasp_locations_to_indices_and_offsets_and_xpos(phy, locs)
            # phy_ik, _ = create_eq_and_sim_ik(phy, self.tool_names, self.gripper_to_world_eq_names,
            #                                  candidate_is_grasping, candidate_idx, candidate_pos)
            # self.viz.viz(phy_ik, is_planning=True)

            if cost < best_cost:
                best_locs = locs
                best_cost = cost

        if best_cost >= HARD_CONSTRAINT_PENALTY:
            print("No valid different homotopies found")
            return np.array([-1, -1])
        return best_locs

    def get_h_signature(self, phy, initial_rope_points):
        """
        The H-signature is a vector the uniquely defines the homotopy class of the current state. The state involves
        both the gripper and arm positions, the rope configuration, and the obstacles. The h_equal() function can be
        used to compare if two states are the same homotopy class (AKA homologous).
        Args:
            phy: The full physics state
            initial_rope_points: The rope points. Passed separately because the IK solver modifies the rope state
                 in a stupid way that we don't want.

        Returns:
            The h-signature of the current state and the loops and tool positions that were used to compute it.

        """
        # First construct a graph based on the current constraints/grasps
        # The nodes in the graph are:
        #  1. points on the rope fixed in the world (hard-coded based on an EQ with "attach" in its name.)
        #  2. base of the robot
        #  3. grasping grippers
        graph = nx.DiGraph()
        base_xpos = phy.d.body('val_base').xpos
        graph.add_node('b', loc=-1, xpos=base_xpos, point_idx=-1)  # base of the robot. -1 is a dummy value
        rope_length = get_rope_length(phy)
        attach_i = 0
        for eq_idx in range(phy.m.neq):
            eq = phy.m.eq(eq_idx)
            if eq.active:
                body_idx = int(eq.obj2id)
                offset = eq.data[3]
                xmat = phy.d.xmat[body_idx].reshape(3, 3)
                xpos = np.squeeze(phy.d.xpos[body_idx] + xmat[:, 0] * offset)
                loc = float(grasp_indices_to_locations(phy.o.rope.body_indices, body_idx) + (offset / rope_length))
                point_idx = body_idx - phy.o.rope.body_indices[0]
                if "attach" in eq.name:
                    graph.add_node(f'a{attach_i}', loc=loc, xpos=xpos, point_idx=point_idx)
                    attach_i += 1
                elif eq.name == 'left':  # FIXME: hardcoded gripper names
                    graph.add_node(f'g0', loc=loc, xpos=xpos, point_idx=point_idx)
                elif eq.name == 'right':
                    graph.add_node(f'g1', loc=loc, xpos=xpos, point_idx=point_idx)

        while True:
            # Edges in the graph are added based on the following rules. All edges are bidirectional.
            #  1. base can connect to all other nodes
            #  2. grippers and connect to attach nodes
            #  3. grippers can connect to each other
            # Attach nodes cannot connect to each other, since loops between attach points cannot be effected by
            # regrasping, so for the sake of regrasp planning they are irrelevant.
            # In addition to these general rules, we also disallow edges between nodes which pass "through" other nodes via
            # the rope. Consider nodes A, B, and C represent attach or grasped points on the rope. Each of these nodes
            # has a location on the rope between 0 and 1, for instance A=0.2, B=0.5, C=0.8. Then we disallow edges between
            # A and C, since that would pass through B.
            #
            # Each edge also has an associated "edge_path" which connects the two xpos's for those nodes.
            # The points are based on the "edges" in the cycle, but the path itself differs depending on the type of edge.
            # For a "b-g" edge, the points come from FK.
            # For a "g-a" or "g-g" edge, the points come from the rope.
            # Specifically, we use the positions of the rope bodies which are between the body indices,
            # which we get from the EQ constraints. The order of the bodies depends on the order of the edge,
            # i.e "g0->a" vs "a->g0". Finally, an edge like "b->a" is simply the robot base position and the "a" position.
            # order still matters.
            arm_points = [parents_points(phy.m, phy.d, tool_body) for tool_body in self.tool_bodies]
            for i in graph.nodes:
                for j in graph.nodes:
                    if i == j:
                        continue
                    # Check if the edge passes through any other nodes
                    if passes_through(graph, i, j):
                        continue
                    i_xpos = graph.nodes[i]['xpos']
                    j_xpos = graph.nodes[j]['xpos']
                    i_point_idx = graph.nodes[i]['point_idx']
                    j_point_idx = graph.nodes[j]['point_idx']
                    if i == 'b' and 'g' in j:
                        arm_idx = int(j[1])
                        graph.add_edge(i, j, edge_path=arm_points[arm_idx][::-1])  # ::-1 to get base -> gripper
                    elif 'g' in i and j == 'b':
                        arm_idx = int(i[1])
                        graph.add_edge(i, j, edge_path=arm_points[arm_idx])
                    elif i == 'b' and 'a' in j:
                        graph.add_edge(i, j, edge_path=np.stack([i_xpos, floorify(i_xpos), floorify(j_xpos), j_xpos]))
                    elif 'a' in i and j == 'b':
                        graph.add_edge(i, j, edge_path=np.stack([i_xpos, floorify(i_xpos), floorify(j_xpos), j_xpos]))
                    elif 'g' in i and 'g' in j:
                        edge_rope_points = initial_rope_points[from_to(i_point_idx, j_point_idx)]
                        graph.add_edge(i, j, edge_path=np.stack([i_xpos, *edge_rope_points, j_xpos]))
                    elif 'g' in i and 'a' in j:
                        edge_rope_points = initial_rope_points[from_to(i_point_idx, j_point_idx)]
                        graph.add_edge(i, j, edge_path=np.stack([i_xpos, *edge_rope_points, j_xpos]))
                    elif 'a' in i and 'g' in j:
                        edge_rope_points = initial_rope_points[from_to(i_point_idx, j_point_idx)]
                        graph.add_edge(i, j, edge_path=np.stack([i_xpos, *edge_rope_points, j_xpos]))

            # import matplotlib.pyplot as plt
            # loc_labels = nx.get_node_attributes(graph, 'loc')
            # pos_labels = nx.get_node_attributes(graph, 'xpos')
            # combined_labels = {k: f"{k}: {loc_labels[k]:.2f} {pos_labels[k]}" for k in graph.nodes}
            # nx.draw(graph, labels=combined_labels, node_size=1000)
            # plt.margins(x=0.4)
            # plt.show()

            valid_cycles = []
            for cycle in nx.simple_cycles(graph, length_bound=3):
                # Cycle must be length 3 and contain the robot base.
                # There could be a cycle like ['g0', 'g1', 'a'], but we don't need to consider that
                # Since that only involves the points on the rope, so it will always be in the same homotopy class
                # Also, we filter out ones that are the same as previous ones up to ordering (aka edges have no direction)
                # We used a DiGraph initially because edge direction does matter for the edge_path, but for the cycle
                # detection we don't care about direction.
                if len(cycle) == 3 and 'b' in cycle and new_cycle(cycle, valid_cycles):
                    valid_cycles.append(cycle)

            if len(valid_cycles) == 0:
                return None, -999  # No valid cycles, so we give it a bogus h value

            loops = []
            for valid_cycle in valid_cycles:
                valid_cycle = valid_cycle + [valid_cycle[0]]
                loop = []
                for edge in pairwise(valid_cycle):
                    edge_path = graph.get_edge_data(*edge)['edge_path']
                    loop.extend(edge_path)
                loop.append(loop[0])
                loop = np.stack(loop)
                loops.append(loop)

            for l in loops:
                rr.log_line_strip('candidate_loop', l)

            removed_node = False
            for loop, cycle in zip(loops, valid_cycles):
                # If the loop is between grippers, and the h-signature is 0, then delete ones of the two gripper
                # nodes and re-run the algorithm
                edge = has_gripper_gripper_edge(cycle)
                if edge is not None:
                    h = get_h_signature(loop, self.skeletons)
                    if np.count_nonzero(h) == 0:
                        # arbitrarily choose the "larger" node in the edge as the one we remove.
                        graph.remove_node(max(*edge))
                        removed_node = True
                        break
            if removed_node:
                continue

            h = np.array([get_h_signature(loop, self.skeletons) for loop in loops])

            return loops, h


def passes_through(graph, i, j):
    for k in graph.nodes:
        if k == i or k == j:
            continue
        if k == 'b' or i == 'b' or j == 'b':
            continue
        i_loc = graph.nodes[i]['loc']
        j_loc = graph.nodes[j]['loc']
        k_loc = graph.nodes[k]['loc']
        lower = min(i_loc, j_loc)
        upper = max(i_loc, j_loc)
        if lower < k_loc < upper:
            return True
    return False


def new_cycle(cycle, valid_cycles):
    for valid_cycle in valid_cycles:
        if set(cycle) == set(valid_cycle):
            return False
    return True


def floorify(xpos):
    xpos = xpos.copy()
    xpos[2] = -1
    return xpos


def has_gripper_gripper_edge(loop):
    # pairwise ( + ) is a way to iterate over a list in pairs but ensure we loop back around
    for e1, e2 in pairwise(loop + loop[0:1]):
        if 'g' in e1 and 'g' in e2:
            return e1, e2
    return None


def create_eq_and_sim_ik(phy, tool_names, gripper_to_world_eq_names, candidate_is_grasping, candidate_idx,
                         candidate_pos):
    phy_ik = phy.copy_all()
    grasp_eqs = get_grasp_eqs(phy_ik.m)
    ik_targets = {}
    for i in range(hp['n_g']):
        tool_name = tool_names[i]
        gripper_to_world_eq_name = gripper_to_world_eq_names[i]
        eq = grasp_eqs[i]
        # Set eq_active and qpos depending on the grasp
        if candidate_is_grasping[i]:
            eq.active = 1
            eq.obj2id = candidate_idx[i]
            ik_targets[tool_name] = candidate_pos[i] - IK_OFFSET
            gripper_to_world_eq = phy_ik.m.eq(gripper_to_world_eq_name)
            gripper_to_world_eq.active = 1
            gripper_to_world_eq.data[3:6] = candidate_pos[i]
        else:
            eq.active = 0
    # Estimate reachability using eq constraints in mujoco
    reached = eq_sim_ik(tool_names, candidate_is_grasping, candidate_pos, phy_ik, viz=None)
    return phy_ik, reached


def make_full_locs(locs_where_grasping, is_grasping):
    candidate_locs = []
    j = 0
    for is_grasping_i in is_grasping:
        if is_grasping_i:
            candidate_locs.append(locs_where_grasping[j])
            j += 1
        else:
            candidate_locs.append(-1)
    candidate_locs = np.array(candidate_locs)
    return candidate_locs


def sln_to_locs(best_sln, candidate_is_grasping):
    locs = []
    best_sln = list(best_sln.values())
    j = 0
    for is_grasping_i in candidate_is_grasping:
        if is_grasping_i:
            locs.append(best_sln[j])
            j += 1
        else:
            locs.append(-1)
    locs = np.array(locs)
    return locs
