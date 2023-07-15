"""
This file contains the HomotopyChecker class, which is used to compare two states to see if they are homologous.
This considers both true homotopy and first-order homotopy.
"""
from typing import Dict

import networkx as nx
import numpy as np
import pysdf_tools
import rerun as rr
from pymjregrasping_cpp import get_first_order_homotopy_points

from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.grasp_conversions import grasp_indices_to_locations
from mjregrasping.homotopy_utils import get_full_h_signature, passes_through, floorify, from_to
from mjregrasping.mujoco_objects import parents_points
from mjregrasping.physics import Physics
from mjregrasping.rope_length import get_rope_length


def get_first_order_different(sdf: pysdf_tools.SignedDistanceField, phy1: Physics, phy2: Physics):
    rope1 = get_rope_points(phy1)
    rope2 = get_rope_points(phy2)
    first_order_sln = get_first_order_homotopy_points(sdf, rope1, rope2, -sdf.GetResolution())
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


def get_full_h_signature_from_phy(skeletons: Dict, phy: Physics):
    graph = create_graph_nodes(phy)
    rope_points = get_rope_points(phy)
    arm_points = get_arm_points(phy)

    return get_full_h_signature(skeletons, graph, rope_points, arm_points)


def get_arm_points(phy: Physics):
    arm_points = [parents_points(phy.m, phy.d, tool_body) for tool_body in phy.o.rd.tool_bodies]
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
