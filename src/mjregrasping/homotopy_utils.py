"""
Functions for computing h-signatures of paths through the environment.
Based on this paper: https://www.roboticsproceedings.org/rss07/p02.pdf
"""
from copy import deepcopy
from typing import Dict

import hjson
import networkx as nx
import numpy as np
from multiset import Multiset
from numpy.linalg import norm

from mjregrasping.geometry import squared_norm
from mjregrasping.homotopy_checker import add_edges


def get_h_signature(path, skeletons: Dict):
    """
    Computes the h-signature of a path, given the skeletons of the obstacles.

    Args:
        path: A path through the environment, as a list of points in 3D.
        skeletons:  A dictionary of skeletons, where the keys are the names of the obstacles and the values are the
            skeletons of the obstacles.

    Returns:

    """
    # Densely discretize the path so that we can integrate the field along it
    path_discretized = discretize_path(path)
    path_deltas = np.diff(path_discretized, axis=0)
    hs = []
    for skeleton in skeletons.values():
        bs = skeleton_field_dir(skeleton, path_discretized[:-1])
        # Integrate the field along the path
        h = np.sum(np.sum(bs * path_deltas, axis=-1), axis=0)
        # round to nearest integer since the output should really either be 0 or 1
        # absolute value because we don't care which "direction" the loop goes through an obstacle
        h = abs(int(h.round(0)))
        hs.append(h)
    return tuple(hs)


def skeleton_field_dir(skeleton, r):
    """
    Computes the field direction at the input points, where the conductor is the skeleton of an obstacle.
    A skeleton is defined by a set of points in 3D, like a line-strip, and can represent only a genus-1 obstacle (donut)
    Assumes μ and I are 1.
    Based on this paper: https://www.roboticsproceedings.org/rss07/p02.pdf

    Variables in my code <--> math in the paper:

        s_prev = s_i^j
        s_next = s_i^j'
        p_prev = p
        p_next = p'

    Args:
        skeleton: [n, 3] the points that define the skeleton
        r: [b, 3] the points at which to compute the field.
    """
    if not np.all(skeleton[0] == skeleton[-1]):
        raise ValueError("Skeleton must be a closed loop! Add the first point to the end.")

    s_prev = skeleton[:-1][None]  # [1, n, 3]
    s_next = skeleton[1:][None]  # [1, n, 3]

    p_prev = s_prev - r[:, None]  # [b, n, 3]
    p_next = s_next - r[:, None]  # [b, n, 3]
    squared_segment_lens = squared_norm(s_next - s_prev, keepdims=True)
    d = np.cross((s_next - s_prev), np.cross(p_next, p_prev)) / squared_segment_lens  # [b, n, 3]

    # bs is a matrix [b, n,3] where each bs[i, j] corresponds to a line segment in the skeleton
    squared_d_lens = squared_norm(d, keepdims=True)
    p_next_lens = norm(p_next, axis=-1, keepdims=True) + 1e-6
    p_prev_lens = norm(p_prev, axis=-1, keepdims=True) + 1e-6

    # Epsilon is added to the denominator to avoid dividing by zero, which would happen for points _on_ the skeleton.
    ε = 1e-6
    d_scale = np.where(squared_d_lens > ε, 1 / (squared_d_lens + ε), 0)

    bs = d_scale * (np.cross(d, p_next) / p_next_lens - np.cross(d, p_prev) / p_prev_lens)

    b = bs.sum(axis=1) / (4 * np.pi)
    return b


def make_ring_skeleton(position, z_axis, radius, delta_angle=0.5):
    angles = np.arange(0, 2 * np.pi, delta_angle)
    angles = np.append(angles, 0)
    zeros = np.zeros_like(angles)
    ones = np.ones_like(angles)
    ring_skeleton_canonical = np.stack([radius * np.cos(angles), radius * np.sin(angles), zeros, ones], -1)
    ring_mat = make_ring_mat(position, z_axis)
    skeleton = (ring_skeleton_canonical @ ring_mat.T)[:, :3]
    return skeleton


def make_ring_mat(ring_position, ring_z_axis):
    """
    Make the 4x4 transform matrix that transforms from the canonical ring frame to the world frame.
    The canonical ring frame has +Z going up out of the ring

    Args:
        ring_position: [3] position of the center of the ring
        ring_z_axis: [3] unit vector pointing up out of the ring

    Returns: [4, 4] transform matrix

    """
    rand = np.random.rand(3)
    # project rand to its component in the plane of the ring
    ring_y_axis = rand - np.dot(rand, ring_z_axis) * ring_z_axis
    ring_y_axis /= np.linalg.norm(ring_y_axis)
    ring_x_axis = np.cross(ring_y_axis, ring_z_axis)
    ring_mat = np.eye(4)
    ring_mat[:3, 0] = ring_x_axis
    ring_mat[:3, 1] = ring_y_axis
    ring_mat[:3, 2] = ring_z_axis
    ring_mat[:3, 3] = ring_position
    return ring_mat


def discretize_path(path, n=1000):
    path_deltas = np.diff(path, axis=0)
    delta_lengths = np.linalg.norm(path_deltas, axis=-1)
    delta_cumsums = np.cumsum(delta_lengths)
    delta_cumsums = np.insert(delta_cumsums, 0, 0)
    total_length = np.sum(delta_lengths)
    ε = 1e-3
    l_s = np.linspace(ε, total_length - ε, n)
    i_s = np.searchsorted(delta_cumsums, l_s)
    discretized_path_points = path[i_s - 1] + (l_s - delta_cumsums[i_s - 1])[:, None] * path_deltas[i_s - 1] / \
                              delta_lengths[i_s - 1][:, None]
    discretized_path_points = np.insert(discretized_path_points, 0, path[0], axis=0)
    return discretized_path_points


def load_skeletons(skeleton_filename):
    with open(skeleton_filename, 'r') as f:
        skeletons = hjson.load(f)
    return {k: np.array(v) for k, v in skeletons.items()}


NO_HOMOTOPY = Multiset([-999])


def get_full_h_signature(skeletons: Dict, graph, rope_points, arm_points):
    """
    The H-signature is a vector the uniquely defines the homotopy class of the current state. The state involves
    both the gripper and arm positions, the rope configuration, and the obstacles. The h_equal() function can be
    used to compare if two states are the same homotopy class (AKA homologous).
    Args:
        skeletons: The skeletons of the obstacles in the environment.
        graph: The graph of the current state. This is a networkx DiGraph, where the nodes are the points on the rope,
            the base of the robot, and the tool bodies of the robot. The edges are the paths between the nodes.
        rope_points: The points on the rope.
        arm_points: points of the arms

    Returns:
        The h-signature of the current state and the loops and tool positions that were used to compute it.

    """
    while True:
        add_edges(graph, rope_points, arm_points)

        skeletons = deepcopy(skeletons)

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
            return NO_HOMOTOPY  # No valid cycles, so we give it a bogus h value

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

        removed_node = False
        for loop, cycle in zip(loops, valid_cycles):
            # If the loop is between grippers, and the h-signature is 0, then delete ones of the two gripper
            # nodes and re-run the algorithm
            edge = has_gripper_gripper_edge(cycle)
            if edge is not None:
                h = get_h_signature(loop, skeletons)
                if np.count_nonzero(h) == 0:
                    # arbitrarily choose the "larger" node in the edge as the one we remove.
                    graph.remove_node(max(*edge))
                    removed_node = True
                    break
        if removed_node:
            continue

        h = Multiset([get_h_signature(loop, skeletons) for loop in loops])

        return h, loops


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


def pairwise(x):
    return zip(x[:-1], x[1:])


def from_to(i, j):
    """ Inclusive of both i and j """
    if i < j:
        return range(i + 1, j + 1)
    else:
        return range(i, j, -1)
