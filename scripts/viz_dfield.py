import mujoco
import numpy as np

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from geometry_msgs.msg import Point
from mjregrasping.basic_3d_pose_marker import Basic3DPoseInteractiveMarker
from mjregrasping.dijsktra_field import make_dfield
from mjregrasping.rviz import MjRViz
from ros_numpy import numpify, msgify


def main():
    rospy.init_node("viz_dfield")

    goal_point = np.array([0.73, 0.04, 1.25])
    res = 0.03
    extents_2d = np.array([[0.6, 1.4], [-0.7, 0.4], [0.2, 1.3]])
    xml_path = "models/untangle_scene.xml"
    m = mujoco.MjModel.from_xml_path(xml_path)
    tfw = TF2Wrapper()
    mjviz = MjRViz(xml_path, tfw)
    d = mujoco.MjData(m)
    dfield = make_dfield(m, d, extents_2d, res, goal_point)
    i = Basic3DPoseInteractiveMarker(0.7, 0, 0.7, 'sphere')
    print("Ready!")

    while not rospy.is_shutdown():
        mjviz.viz(m, d, is_planning=True)

        start_point_msg = i.get_pose().position
        start_point = numpify(start_point_msg)
        for idx in range(25):
            grad = dfield.get_grad(start_point)
            sdf = dfield.get_vg(start_point)
            end_point = start_point + -grad * 0.02
            end_point_msg = msgify(Point, end_point)
            r = 1 - np.clip(3 * sdf + 0.05, 0, 1)
            # points in the direction of the dfield gradient
            start_point_msg = msgify(Point, start_point)
            dfield.viz(end_point_msg, start_point_msg, r, idx=idx)
            rospy.sleep(0.01)
            start_point = end_point

        rospy.sleep(0.1)


if __name__ == '__main__':
    main()
