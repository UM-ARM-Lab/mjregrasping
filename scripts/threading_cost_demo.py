from time import perf_counter

import numpy as np

import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    pub = rospy.Publisher('markers', MarkerArray, queue_size=10)

    rospy.init_node('marker_publisher')

    mu = 1.0

    R = 0.5  # radius
    I = 50.0  # current

    viz_ring(pub, R)

    ys = np.arange(-1.0, 1.0, 0.05)
    zs = np.linspace(-0.5, 0.5, 10)
    idx = 0
    for y in ys:
        for z in zs:
            p = np.array([0, y, z])
            b = compute_threading_cost(I, R, mu, p)
            viz_arrow(pub, p, b, idx=idx)
            idx += 1

    # last_p = None
    # p = np.array([0, 1.5, 0.05])
    # while True:
    #     if last_p is not None and np.linalg.norm(last_p) < 0.01:
    #         break
    #
    #
    #     viz_arrow(pub, p, b)
    #
    #     last_p = p
    #
    #     p = p - b


def compute_threading_cost(I, R, mu, p):
    # integrate
    delta_angle = 0.1
    angles = np.arange(0, 2 * np.pi, delta_angle)
    # point along the ring, under the assumption the ring is in the x/y plane
    zeros = np.zeros_like(angles)
    x = np.stack([R * np.cos(angles), R * np.sin(angles), zeros], -1)
    dx = np.stack([-np.sin(angles), np.cos(angles), zeros], -1) * delta_angle
    dx = dx / np.linalg.norm(dx, axis=-1, keepdims=True)  # normalize
    db = I * R * np.cross(dx, (p - x)) / np.linalg.norm(p - x, axis=-1, keepdims=True)**3
    b = np.sum(db, axis=0)
    b *= mu / (4 * np.pi)
    return b


def viz_arrow(pub, start, direction, idx=0, r=1, b=0, g=0):
    arrow_msg = Marker()
    arrow_msg.header.frame_id = "world"
    arrow_msg.header.stamp = rospy.Time.now()
    arrow_msg.ns = "field"
    arrow_msg.id = idx
    arrow_msg.type = Marker.ARROW
    arrow_msg.action = Marker.ADD
    arrow_msg.pose.orientation.w = 1.0
    arrow_msg.scale.x = 0.005
    arrow_msg.scale.y = 0.01
    arrow_msg.scale.z = 0.01
    arrow_msg.color.a = 1.0
    arrow_msg.color.r = r
    arrow_msg.color.g = g
    arrow_msg.color.b = b
    d = np.linalg.norm(direction)
    max_d = 0.01
    if d > max_d:
        direction = direction / d * max_d
    end = start + direction * 5
    arrow_msg.points.append(Point(start[0], start[1], start[2]))
    arrow_msg.points.append(Point(end[0], end[1], end[2]))
    markers_msg = MarkerArray()
    markers_msg.markers.append(arrow_msg)

    for _ in range(3):
        pub.publish(markers_msg)
        rospy.sleep(0.001)

def viz_ring(pub, radius, idx=0):
    ring_msg = Marker()
    ring_msg.header.frame_id = "world"
    ring_msg.header.stamp = rospy.Time.now()
    ring_msg.ns = "ring"
    ring_msg.id = idx
    ring_msg.type = Marker.LINE_STRIP
    ring_msg.action = Marker.ADD
    ring_msg.pose.orientation.w = 1.0
    ring_msg.scale.x = 0.05
    ring_msg.color.a = 1.0
    ring_msg.color.g = 1.0
    for angle in np.arange(0, 3 * np.pi, 0.1):  # 3 pi to ensure it overlaps
        p = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
        ring_msg.points.append(Point(p[0], p[1], p[2]))
    markers_msg = MarkerArray()
    markers_msg.markers.append(ring_msg)

    for _ in range(10):
        pub.publish(markers_msg)
        rospy.sleep(0.1)


if __name__ == '__main__':
    main()
