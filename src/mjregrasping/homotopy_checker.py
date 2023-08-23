"""
This file contains the HomotopyChecker class, which is used to compare two states to see if they are homologous.
This considers both true homotopy and first-order homotopy.
"""
from copy import deepcopy
from enum import Enum, auto
from typing import Dict

import networkx as nx
import numpy as np
import rerun as rr
from multiset import Multiset
from pymjregrasping_cpp import get_first_order_homotopy_points

from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.grasp_conversions import grasp_indices_to_locations
from mjregrasping.homotopy_utils import passes_through, floorify, from_to, check_new_cycle, NO_HOMOTOPY, pairwise, \
    has_gripper_gripper_edge, get_h_signature
from mjregrasping.mujoco_objects import parents_points
from mjregrasping.physics import Physics
from mjregrasping.rope_length import get_rope_length


class AllowablePenetration(Enum):
    NONE = auto()
    HALF_CELL = auto()
    FULL_CELL = auto()


class CollisionChecker:
    """
    Defines the simple interface needed for collision checking of points in 3D.
    """

    def __init__(self):
        pass

    def is_collision(self, point, allowable_penetration: AllowablePenetration = AllowablePenetration.NONE):
        """
        Returns True if the point is in collision, False otherwise.
        """
        raise NotImplementedError()

    def get_resolution(self):
        """

        Returns: The resolution of the collision checker. This is the minimum distance between two points that the
        collision checker can distinguish between. This is common for voxelgrid or SDF representations, but also sets
        the precision of collision checking in fully continuous collision checkers.

        """
        raise NotImplementedError()


def get_first_order_different(collision_checker: CollisionChecker, phy1: Physics, phy2: Physics):
    rope1 = get_rope_points(phy1)
    rope2 = get_rope_points(phy2)

    def _in_collision(p):
        return collision_checker.is_collision(p, allowable_penetration=AllowablePenetration.FULL_CELL)

    first_order_sln = get_first_order_homotopy_points(_in_collision, rope1, rope2)
    first_order_different = len(first_order_sln) == 0
    return first_order_different


def get_true_homotopy_different(skeletons: Dict, phy1: Physics, phy2: Physics, log_loops=False):
    h1, loops1 = get_full_h_signature_from_phy(skeletons, phy1)
    h2, loops2 = get_full_h_signature_from_phy(skeletons, phy2)
    true_homotopy_different = (h1 != h2)

    if log_loops:
        rr.log_cleared(f'loops1', recursive=True)
        for i, l in enumerate(loops1):
            rr.log_line_strip(f'loops1/{i}', l, stroke_width=0.02)
        rr.log_cleared(f'loops2', recursive=True)
        for i, l in enumerate(loops2):
            rr.log_line_strip(f'loops2/{i}', l, stroke_width=0.02)

    return true_homotopy_different


DEFAULT_COLLAPSE_EMPTY_GRIPPER_CYCLES = True
GRIPPER_IDS_IN_H_SIGNATURE = False


def get_full_h_signature_from_phy(skeletons: Dict, phy: Physics,
                                  collapse_empty_gripper_cycles=DEFAULT_COLLAPSE_EMPTY_GRIPPER_CYCLES,
                                  gripper_ids_in_h_signature=GRIPPER_IDS_IN_H_SIGNATURE):
    graph = create_graph_nodes(phy)
    rope_points = get_rope_points(phy)
    arm_points = get_arm_points(phy)

    return get_full_h_signature(skeletons, graph, rope_points, arm_points,
                                collapse_empty_gripper_cycles,
                                gripper_ids_in_h_signature)


def get_loops_from_phy(phy):
    graph = create_graph_nodes(phy)
    rope_points = get_rope_points(phy)
    arm_points = get_arm_points(phy)

    add_edges(graph, rope_points, arm_points)

    valid_cycles = get_valid_cycles(graph)

    if len(valid_cycles) == 0:
        raise ValueError("No valid cycles")

    return get_loops(graph, valid_cycles)


def get_arm_points(phy: Physics):
    arm_points = []
    for tool_body, tool_site in zip(phy.o.rd.tool_bodies, phy.o.rd.tool_sites):
        arm_points_i = parents_points(phy.m, phy.d, tool_body)
        arm_points_i = np.insert(arm_points_i, 0, phy.d.site(tool_site).xpos, 0)
        arm_points.append(arm_points_i)
    return arm_points


def create_graph_nodes(phy: Physics):
    # First construct a graph based on the current constraints/grasps
    # The nodes in the graph are:
    #  1. points on the rope fixed in the world (hard-coded based on an EQ with "attach" in its name.)
    #  2. base of the robot
    #  3. grasping grippers
    graph = nx.DiGraph()
    base_xpos = phy.d.body(phy.o.rd.base_link).xpos
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
            if eq.name in phy.o.rd.rope_grasp_eqs:
                eq_name_idx = phy.o.rd.rope_grasp_eqs.index(eq.name)
                graph.add_node(f'g{eq_name_idx}', loc=loc, xpos=xpos, point_idx=point_idx)
    return graph


def add_edges(graph, rope_points, arm_points):
    """
    Edges in the graph are added based on the following rules. All edges are bidirectional.
     1. base can connect to all other nodes
     2. grippers and connect to attach nodes
     3. grippers can connect to each other
    Attach nodes cannot connect to each other, since loops between attach points cannot be effected by
    regrasping, so for the sake of regrasp planning they are irrelevant.
    In addition to these general rules, we also disallow edges between nodes which pass "through" other nodes via
    the rope. Consider nodes A, B, and C represent attach or grasped points on the rope. Each of these nodes
    has a location on the rope between 0 and 1, for instance A=0.2, B=0.5, C=0.8. Then we disallow edges between
    A and C, since that would pass through B.

    Each edge also has an associated "edge_path" which connects the two xpos's for those nodes.
    The points are based on the "edges" in the cycle, but the path itself differs depending on the type of edge.
    For a "b-g" edge, the points come from FK.
    For a "g-a" or "g-g" edge, the points come from the rope.
    Specifically, we use the positions of the rope bodies which are between the body indices,
    which we get from the EQ constraints. The order of the bodies depends on the order of the edge,
    i.e "g0->a" vs "a->g0". Finally, an edge like "b->a" is simply the robot base position and the "a" position.
    order still matters.

    Args:
        graph: networkx Graph representing the robot and fixed points 'connect' to the object
        rope_points: points representing the object (rope) [n_rope_points, 3]
        arm_points: points tracing the robot arms [n_arms, n_points, 3]

    Returns:

    """
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
                graph.add_edge(i, j,
                               edge_path=np.stack([i_xpos, floorify(i_xpos), floorify(j_xpos), j_xpos]))
            elif 'a' in i and j == 'b':
                graph.add_edge(i, j,
                               edge_path=np.stack([i_xpos, floorify(i_xpos), floorify(j_xpos), j_xpos]))
            elif 'g' in i and 'g' in j:
                edge_rope_points = rope_points[from_to(i_point_idx, j_point_idx)]
                graph.add_edge(i, j, edge_path=np.stack([i_xpos, *edge_rope_points, j_xpos]))
            elif 'g' in i and 'a' in j:
                edge_rope_points = rope_points[from_to(i_point_idx, j_point_idx)]
                graph.add_edge(i, j, edge_path=np.stack([i_xpos, *edge_rope_points, j_xpos]))
            elif 'a' in i and 'g' in j:
                edge_rope_points = rope_points[from_to(i_point_idx, j_point_idx)]
                graph.add_edge(i, j, edge_path=np.stack([i_xpos, *edge_rope_points, j_xpos]))


def get_full_h_signature(skeletons: Dict, graph, rope_points, arm_points,
                         collapse_empty_gripper_cycles, gripper_ids_in_h_signature):
    """
    This function computes the full h-signature of the current state.
    Two states are homologous if and only if they have the same h-signature.

    The H-signature uniquely defines the homotopy class of the current state, and is represented as  a multi-set.
    # The state involves the arms, the object (rope) configuration, and the obstacles (skeletons).
    Args:
        skeletons: The skeletons of the obstacles in the environment.
        graph: The graph of the current state. This is a networkx DiGraph, where the nodes are the points on the rope,
            the base of the robot, and the tool bodies of the robot. The edges are the paths between the nodes.
        rope_points: The points on the rope.
        arm_points: points of the arms
        collapse_empty_gripper_cycles: If True, then if there is a cycle between two grippers and the h-signature of
            the cycle is 0, then we remove one of the grippers and re-run the algorithm. You probably want to set this
            to False if you are planning hand-over-hand motions.
        gripper_ids_in_h_signature: If True, then the h-signature will include the ids of the grippers. You probably
            want to set this to True if you are planning hand-over-hand motions.

    Returns:
        The h-signature of the current state, and the loops and tool positions that were used to compute it.

    """
    while True:
        add_edges(graph, rope_points, arm_points)

        skeletons = deepcopy(skeletons)

        valid_cycles = get_valid_cycles(graph)

        if len(valid_cycles) == 0:
            return NO_HOMOTOPY, []  # No valid cycles, so we give it a bogus h value

        loop_ids, loops = get_loops(graph, valid_cycles)

        if collapse_empty_gripper_cycles:
            removed_node = False
            for loop, cycle in zip(loops, valid_cycles):
                # If the loop is between grippers, and the h-signature is 0, then delete ones of the two gripper
                # nodes and re-run the algorithm
                gg_edge = has_gripper_gripper_edge(cycle)
                if gg_edge is not None:
                    h = get_h_signature(loop, skeletons)
                    if np.count_nonzero(h) == 0:
                        # arbitrarily choose the "larger" node in the edge as the one we remove.
                        graph.remove_node(max(*gg_edge))
                        removed_node = True
                        break
            if removed_node:
                continue

        h = [get_h_signature(loop, skeletons) for loop in loops]
        if gripper_ids_in_h_signature:
            # Add the gripper ids to the h-signature
            h = [(loop_id,) + h_i for loop_id, h_i in zip(loop_ids, h)]
        h = Multiset(h)

        return h, loops


def get_loops(graph, valid_cycles):
    loops = []
    loop_ids = []
    for valid_cycle in valid_cycles:
        # Close the cycle by adding the first node to the end,
        # so that we can get the edge_path of edge from the last node back to the first node.
        valid_cycle = valid_cycle + [valid_cycle[0]]
        loop = []
        for edge in pairwise(valid_cycle):
            edge_path = graph.get_edge_data(*edge)['edge_path']
            loop.extend(edge_path)
        loop.append(loop[0])
        loop = np.stack(loop)
        loop_id = ','.join(sorted(valid_cycle[:-1]))
        loops.append(loop)
        loop_ids.append(loop_id)
    return loop_ids, loops


def get_valid_cycles(graph):
    valid_cycles = []
    for cycle in nx.simple_cycles(graph, length_bound=3):
        # Cycle must be length 3 and contain the base.
        # Also, we filter out ones that are the same as previous ones up to ordering (aka edges have no direction)
        # We used a DiGraph initially because edge direction does matter for the edge_path, but for the cycle
        # detection we don't care about direction.
        if len(cycle) == 3 and 'b' in cycle and check_new_cycle(cycle, valid_cycles):
            # ensure all cycles start with b
            while True:
                if cycle[0] == 'b':
                    break
                cycle = cycle[1:] + cycle[:1]
            valid_cycles.append(cycle)
    return valid_cycles


def get_h_signature_for_goal(skeletons: Dict, rope_points, goal_rope_points):
    """
    Compares rope_points to goal_rope_points, using the skeletons to compute the h-signature.

    Args:
        skeletons: A dictionary of skeletons, where the keys are the names of the obstacles and the values are the
            skeletons of the obstacles.
        rope_points: The points representing the rope in the current state, which we want to compare to the goal
        goal_rope_points: The points representing the rope in the goal state.

    Returns: a list of h-signatures, one for each skeleton.

    """
    # Create a closed loop by connecting the start and end points of the two rope states
    loop = np.concatenate((rope_points, goal_rope_points[::-1], rope_points[0][None]))
    h = get_h_signature(loop, skeletons)
    return h


def compare_h_signature_to_goal(skeletons: Dict, rope_points, goal_rope_points):
    """
    Returns True if the rope_points and goal_rope_points are in the same homotopy class.
    ASSUMPTION: the start and end points of rope_points and goal_rope_points are close or the same.
    """
    h = get_h_signature_for_goal(skeletons, rope_points, goal_rope_points)
    zero_h = (0,) * len(skeletons)
    return h == zero_h


def compare_to_goal(skeletons: Dict, rope_points, goal_rope_points, tol=0.05):
    """
    Returns True if the rope_points and goal_rope_points are in the same homotopy class AND have close start/end points.
    tol is the allowable distance between the start/end points, in meters.
    """
    h_same = compare_h_signature_to_goal(skeletons, rope_points, goal_rope_points)
    start_same = np.linalg.norm(rope_points[0] - goal_rope_points[0]) < tol
    end_same = np.linalg.norm(rope_points[-1] - goal_rope_points[-1]) < tol
    return h_same and start_same and end_same
