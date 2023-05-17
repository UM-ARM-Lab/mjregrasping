import pickle
from pathlib import Path

import numpy as np

import rospy
from mjregrasping.physics import Physics
from mjregrasping.voxelgrid import VoxelGrid, point_to_idx
from visualization_msgs.msg import Marker, MarkerArray


class DField:

    def __init__(self, vg: VoxelGrid, goal: np.ndarray):
        self.vg = vg
        self.goal = goal
        self.pub = rospy.Publisher("dfield", MarkerArray, queue_size=10)

        offsets = np.array([
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ])
        visited = set()
        goal_i = point_to_idx(goal, vg.origin_point, vg.res)
        queue = [goal_i]
        self.dijsktra_field = np.ones_like(vg.vg) * 1000
        self.dijsktra_field[tuple(goal_i)] = 0
        self.grad = np.zeros([vg.shape[0], vg.shape[1], vg.shape[2], 3])

        while len(queue) > 0:
            i = tuple(queue.pop(0))
            for offset in offsets:
                neighbor = np.array(i) + offset
                if np.any(neighbor < 0) or np.any(neighbor >= self.dijsktra_field.shape):
                    continue
                if tuple(neighbor) in visited:
                    continue
                if self.vg.vg[tuple(neighbor)] == 1.0:
                    continue
                new_cost = self.dijsktra_field[i] + 1
                if new_cost < self.dijsktra_field[tuple(neighbor)]:
                    self.dijsktra_field[tuple(neighbor)] = new_cost
                    self.grad[tuple(neighbor)] = offset
                    queue.append(neighbor)
            visited.add(i)

    def get_grad(self, p):
        i_in_bounds = self.p_to_i_in_bounds(p)
        return self.grad[tuple(i_in_bounds)]

    def get_costs(self, points):
        i_in_bounds = self.p_to_i_in_bounds(points)
        costs = self.dijsktra_field[i_in_bounds[..., 0], i_in_bounds[..., 1], i_in_bounds[..., 2]]
        return costs

    def get_vg(self, p):
        i_in_bounds = self.p_to_i_in_bounds(p)
        return self.vg.vg[tuple(i_in_bounds)]

    def p_to_i_in_bounds(self, p):
        i = point_to_idx(p, self.vg.origin_point, self.vg.res)
        i_in_bounds = self.clip(i)
        return i_in_bounds

    def clip(self, i):
        i_in_bounds = np.clip(i, [0, 0, 0], self.vg.shape - 1)
        return i_in_bounds

    def viz(self, end_point_msg, start_point_msg, r, idx):
        markers = make_arrow_marker(end_point_msg, start_point_msg, r, idx)
        self.pub.publish(markers)

    def __getstate__(self):
        # omit the publisher
        state = self.__dict__.copy()
        del state['pub']
        return state

    def __setstate__(self, state):
        # restore the publisher
        self.__dict__.update(state)
        self.pub = rospy.Publisher("dfield", MarkerArray, queue_size=10)


def make_arrow_marker(end_point_msg, start_point_msg, r, idx):
    marker = Marker()
    marker.id = idx
    marker.header.frame_id = "world"
    marker.type = marker.ARROW
    marker.action = marker.ADD
    marker.scale.x = 0.005
    marker.scale.y = 0.01
    marker.scale.z = 0.01
    marker.color.a = 1.0
    marker.color.r = r
    marker.pose.orientation.w = 1.0
    marker.points.append(start_point_msg)
    marker.points.append(end_point_msg)
    markers = MarkerArray()
    markers.markers.append(marker)
    return markers


def make_dfield(phy: Physics, extents_2d, res, goal_point):
    from time import perf_counter
    print("Making dfield...")
    t0 = perf_counter()
    dfield = DField(VoxelGrid(phy, res, extents_2d), goal_point)
    print(f'dfield took {perf_counter() - t0:.1f}s to make.')
    return dfield


def save_load_dfield(phy, goal_point):
    dfield_path = Path("models/dfield.pkl")
    if dfield_path.exists():
        with dfield_path.open('rb') as f:
            dfield = pickle.load(f)
    else:
        res = 0.02
        extents_2d = np.array([[0.6, 1.4], [-0.7, 0.4], [0.2, 1.3]])
        dfield = make_dfield(phy, extents_2d, res, goal_point)
        with dfield_path.open('wb') as f:
            pickle.dump(dfield, f)
    return dfield
