import numpy as np

import rospy
from geometry_msgs.msg import Point
from mjregrasping.goals import make_ring_mat
from visualization_msgs.msg import MarkerArray, Marker


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    pub = rospy.Publisher('markers', MarkerArray, queue_size=10)

    rospy.init_node('marker_publisher')

    mu = 1.0

    ring_position = np.array([0, 1, 1])
    ring_z_axis = np.array([0, 1, 0])
    R = 0.5  # radius
    I = 50.0  # current

    viz_ring(pub, ring_position, ring_z_axis, R)

    xs = np.arange(-1, 1, 0.25)
    ys = np.arange(0.0, 2.0, 0.1)
    zs = np.linspace(0.0, 2.0, 10)
    idx = 0
    for x in xs:
        for y in ys:
            for z in zs:
                p = np.array([x, y, z])
                b = compute_threading_cost(ring_position, ring_z_axis, I, R, mu, p)
                viz_arrow(pub, p, b, idx=idx)
                idx += 1



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


if __name__ == '__main__':
    main()
