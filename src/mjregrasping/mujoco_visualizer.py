from copy import deepcopy
from typing import Optional

import matplotlib.cm as cm
import mujoco
import numpy as np
import transformations
from matplotlib.colors import to_rgba
from mujoco import mju_str2Type, mju_mat2Quat, mjtGeom, mj_id2name

import ros_numpy
import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from geometry_msgs.msg import Point
from mjregrasping.my_transforms import matrix_dist, pos_mat_to_matrix
from mjregrasping.my_transforms import np_wxyz_to_xyzw
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker


class RVizPublishers:

    def __init__(self, tfw: TF2Wrapper):
        self.tfw = tfw
        self.state = rospy.Publisher("state_viz", MarkerArray, queue_size=10)
        self.action = rospy.Publisher("action_viz", MarkerArray, queue_size=10)
        self.ee_path = rospy.Publisher("ee_path", MarkerArray, queue_size=10)
        self.goal = rospy.Publisher("goal_markers", MarkerArray, queue_size=10)


class MujocoVisualizer:
    def __init__(self, tfw: Optional[TF2Wrapper] = None):
        self.tfw = tfw
        self.eq_constraints_pub = rospy.Publisher(
            "eq_constraints", MarkerArray, queue_size=1000, latch=False
        )
        self.pub = rospy.Publisher('all', MarkerArray, queue_size=1000, latch=False)

    def viz(self, model, data, alpha=1, idx=0):
        markers_by_entity = {}

        for geom_id in range(model.ngeom):
            geom_bodyid = model.geom_bodyid[geom_id]
            parent_bodyid = geom_bodyid
            parent_names = []
            while True:
                parent_bodyid = model.body_parentid[parent_bodyid]
                parent_name = mj_id2name(model, mju_str2Type("body"), parent_bodyid)
                parent_names.append(parent_name)
                if parent_bodyid == 0:
                    break
            body_name = mj_id2name(model, mju_str2Type("body"), geom_bodyid)
            entity_name = body_name.split("/")[0]
            if entity_name not in markers_by_entity:
                markers_by_entity[entity_name] = MarkerArray()
            geoms_marker_msg = markers_by_entity[entity_name]

            geom_marker_msg = Marker()
            geom_marker_msg.action = Marker.ADD
            geom_marker_msg.header.frame_id = "world"
            geom_marker_msg.ns = "/".join(parent_names)
            geom_marker_msg.id = idx * 10000 + geom_id

            geom_type = model.geom_type[geom_id]
            body_pos = data.xpos[geom_bodyid]
            body_xmat = data.xmat[geom_bodyid]
            body_xquat = np.zeros(4)
            mju_mat2Quat(body_xquat, body_xmat)
            geom_pos = data.geom_xpos[geom_id]
            geom_xmat = data.geom_xmat[geom_id]
            geom_xquat = np.zeros(4)
            mju_mat2Quat(geom_xquat, geom_xmat)
            geom_size = model.geom_size[geom_id]
            geom_rgba = model.geom_rgba[geom_id]
            geom_meshid = model.geom_dataid[geom_id]

            geom_marker_msg.pose.position.x = geom_pos[0]
            geom_marker_msg.pose.position.y = geom_pos[1]
            geom_marker_msg.pose.position.z = geom_pos[2]
            geom_marker_msg.pose.orientation.w = geom_xquat[0]
            geom_marker_msg.pose.orientation.x = geom_xquat[1]
            geom_marker_msg.pose.orientation.y = geom_xquat[2]
            geom_marker_msg.pose.orientation.z = geom_xquat[3]
            geom_marker_msg.color.r = geom_rgba[0]
            geom_marker_msg.color.g = geom_rgba[1]
            geom_marker_msg.color.b = geom_rgba[2]
            geom_marker_msg.color.a = geom_rgba[3] * alpha

            if geom_type == mjtGeom.mjGEOM_BOX:
                geom_marker_msg.type = Marker.CUBE
                geom_marker_msg.scale.x = geom_size[0] * 2
                geom_marker_msg.scale.y = geom_size[1] * 2
                geom_marker_msg.scale.z = geom_size[2] * 2
            elif geom_type == mjtGeom.mjGEOM_CYLINDER:
                geom_marker_msg.type = Marker.CYLINDER
                geom_marker_msg.scale.x = geom_size[0] * 2
                geom_marker_msg.scale.y = geom_size[0] * 2
                geom_marker_msg.scale.z = geom_size[1] * 2
            elif geom_type == mjtGeom.mjGEOM_CAPSULE:
                # FIXME: not accurate, should use 2 spheres and a cylinder?
                geom_marker_msg.type = Marker.CYLINDER
                geom_marker_msg.scale.x = geom_size[0] * 2
                geom_marker_msg.scale.y = geom_size[0] * 2
                geom_marker_msg.scale.z = geom_size[1] * 2

                geom_marker_msg_ball1: Marker = deepcopy(geom_marker_msg)
                geom_marker_msg_ball1.ns = geom_marker_msg.ns + "b1"
                geom_marker_msg_ball1.type = Marker.SPHERE
                geom_marker_msg_ball1.scale.x = geom_size[0] * 2
                geom_marker_msg_ball1.scale.y = geom_size[0] * 2
                geom_marker_msg_ball1.scale.z = geom_size[0] * 2
                ball1_pos_world = np.zeros(3)
                ball1_pos_local = np.array([0, 0, geom_size[1]])
                geom_xquat_neg = np.zeros(4)
                mujoco.mju_negQuat(geom_xquat_neg, geom_xquat)
                mujoco.mju_rotVecQuat(ball1_pos_world, ball1_pos_local, geom_xquat)
                geom_marker_msg_ball1.pose.position.x += ball1_pos_world[0]
                geom_marker_msg_ball1.pose.position.y += ball1_pos_world[1]
                geom_marker_msg_ball1.pose.position.z += ball1_pos_world[2]

                geom_marker_msg_ball2: Marker = deepcopy(geom_marker_msg)
                geom_marker_msg_ball2.ns = geom_marker_msg.ns + "b2"
                geom_marker_msg_ball2.type = Marker.SPHERE
                geom_marker_msg_ball2.scale.x = geom_size[0] * 2
                geom_marker_msg_ball2.scale.y = geom_size[0] * 2
                geom_marker_msg_ball2.scale.z = geom_size[0] * 2
                ball2_pos_world = np.zeros(3)
                ball2_pos_local = np.array([0, 0, -geom_size[1]])
                geom_xquat_neg = np.zeros(4)
                mujoco.mju_negQuat(geom_xquat_neg, geom_xquat)
                mujoco.mju_rotVecQuat(ball2_pos_world, ball2_pos_local, geom_xquat)
                geom_marker_msg_ball2.pose.position.x += ball2_pos_world[0]
                geom_marker_msg_ball2.pose.position.y += ball2_pos_world[1]
                geom_marker_msg_ball2.pose.position.z += ball2_pos_world[2]

                geoms_marker_msg.markers.append(geom_marker_msg_ball1)
                geoms_marker_msg.markers.append(geom_marker_msg_ball2)
            elif geom_type == mjtGeom.mjGEOM_SPHERE:
                geom_marker_msg.type = Marker.SPHERE
                geom_marker_msg.scale.x = geom_size[0] * 2
                geom_marker_msg.scale.y = geom_size[0] * 2
                geom_marker_msg.scale.z = geom_size[0] * 2
            elif geom_type == mjtGeom.mjGEOM_MESH:
                mesh_name = mj_id2name(
                    model, mju_str2Type("mesh"), geom_meshid
                )
                # skip the model prefix, e.g. val/my_mesh
                if '/' in mesh_name:
                    mesh_name = mesh_name.split("/")[1]
                geom_marker_msg.type = Marker.MESH_RESOURCE
                geom_marker_msg.mesh_use_embedded_materials = True
                geom_marker_msg.mesh_resource = (
                    f"package://mjregrasping/models/meshes/{mesh_name}.stl"
                )

                # We use body pos/quat here under the assumption that in the XML, the <geom type="mesh" ... />
                #  has NO POS OR QUAT, but instead that info goes in the <body> tag
                geom_marker_msg.pose.position.x = body_pos[0]
                geom_marker_msg.pose.position.y = body_pos[1]
                geom_marker_msg.pose.position.z = body_pos[2]
                geom_marker_msg.pose.orientation.w = body_xquat[0]
                geom_marker_msg.pose.orientation.x = body_xquat[1]
                geom_marker_msg.pose.orientation.y = body_xquat[2]
                geom_marker_msg.pose.orientation.z = body_xquat[3]

                geom_marker_msg.scale.x = 1
                geom_marker_msg.scale.y = 1
                geom_marker_msg.scale.z = 1
            else:
                rospy.loginfo_once(f"Unsupported geom type {geom_type}")
                continue

            geoms_marker_msg.markers.append(geom_marker_msg)

        for entity_name, geoms_marker_msg in markers_by_entity.items():
            self.pub.publish(geoms_marker_msg)

        # visualize the weld constraints (regardless of whether they are active)
        eq_lines_msg = MarkerArray()
        for eq_id in range(model.neq):
            eq_name = mj_id2name(model, mju_str2Type("equality"), eq_id)
            if eq_name is not None and "weld" in eq_name:
                eq_data = model.eq_data[eq_id]
                eq_rel_pos = eq_data[:3]
                eq_rel_quat = eq_data[3:]
                parent_id = model.eq_obj1id[eq_id]
                parent_pos = data.xpos[parent_id]
                parent_xmat = data.xmat[parent_id]

                child_id = model.eq_obj2id[eq_id]
                child_pos = data.xpos[child_id]
                child_xmat = data.xmat[child_id]

                # compute the error in position & orientation
                # create the 4x4 transform matrix that represents the transformation from parent_bodyid frame to
                parent2eq = transformations.quaternion_matrix(eq_rel_quat)
                parent2eq[:3, 3] = eq_rel_pos
                world2parent = pos_mat_to_matrix(parent_pos, parent_xmat)
                world2child = pos_mat_to_matrix(child_pos, child_xmat)
                world2eq = world2parent @ parent2eq
                eq_value = matrix_dist(world2eq, world2child)

                eq_line_msg = Marker()
                eq_line_msg.action = Marker.ADD
                eq_line_msg.ns = eq_name
                eq_line_msg.type = Marker.LINE_STRIP
                eq_line_msg.header.frame_id = "world"
                eq_line_msg.pose.orientation.w = 1
                eq_line_msg.scale.x = 0.001
                eq_line_msg.color = ColorRGBA(
                    *to_rgba(cm.viridis(eq_value), alpha=alpha)
                )
                eq_line_msg.points.append(Point(*world2eq[:3, 3]))
                eq_line_msg.points.append(Point(*child_pos))
                eq_lines_msg.markers.append(eq_line_msg)
        self.eq_constraints_pub.publish(eq_lines_msg)

        for body_id in range(model.nbody):
            name = mj_id2name(model, mju_str2Type("body"), body_id)
            pos = data.xpos[body_id]
            mat = data.xmat[body_id]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(mat=mat, quat=quat)
            if self.tfw:
                self.tfw.send_transform(
                    translation=pos,
                    quaternion=np_wxyz_to_xyzw(quat),
                    parent="world",
                    child=name + "_body",
                )
        for geom_id in range(model.ngeom):
            name = mj_id2name(model, mju_str2Type("geom"), geom_id)
            pos = data.geom_xpos[geom_id]
            mat = data.geom_xmat[geom_id]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(mat=mat, quat=quat)
            if self.tfw:
                self.tfw.send_transform(
                    translation=pos,
                    quaternion=np_wxyz_to_xyzw(quat),
                    parent="world",
                    child=name + "_geom",
                )
        for site_id in range(model.nsite):
            name = mj_id2name(model, mju_str2Type("site"), site_id)
            pos = data.site_xpos[site_id]
            mat = data.site_xmat[site_id]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(mat=mat, quat=quat)
            if self.tfw:
                self.tfw.send_transform(
                    translation=pos,
                    quaternion=np_wxyz_to_xyzw(quat),
                    parent="world",
                    child=name + "_site",
                )

        weld_parents = []
        for eq_id in range(model.neq):
            eq_name = mj_id2name(model, mju_str2Type("equality"), eq_id)
            if eq_name is not None and "weld" in eq_name:
                eq_data = model.eq_data[eq_id]
                eq_rel_pos = eq_data[:3]
                eq_rel_quat = eq_data[3:]
                parent_id = model.eq_obj1id[eq_id]
                parent_name = mj_id2name(
                    model, mju_str2Type("body"), parent_id
                )

                # only publish one per parent_bodyid
                if parent_name not in weld_parents:
                    weld_parents.append(parent_name)
                    if self.tfw:
                        self.tfw.send_transform(
                            translation=eq_rel_pos,
                            quaternion=np_wxyz_to_xyzw(eq_rel_quat),
                            parent=parent_name + "_body",
                            child=parent_name + "_weld",
                        )


def plot_sphere_rviz(
        pub, position, radius, frame_id="world", color="m", idx=0, label=""
):
    msg = Marker()
    msg.action = Marker.ADD
    msg.type = Marker.SPHERE
    msg.header.frame_id = frame_id
    msg.scale.x = radius * 2
    msg.scale.y = radius * 2
    msg.scale.z = radius * 2
    msg.color = ColorRGBA(*to_rgba(color))
    msg.id = idx
    msg.ns = label
    msg.pose.orientation.w = 1
    msg.pose.position.x = float(position[0])
    msg.pose.position.y = float(position[1])
    msg.pose.position.z = float(position[2])

    array_msg = MarkerArray()
    array_msg.markers.append(msg)
    pub.publish(array_msg)


def plot_point_rviz(pub, position, idx, label, color="g", frame_id="world"):
    msg = Marker()
    msg.action = Marker.ADD
    msg.type = Marker.SPHERE
    msg.header.frame_id = frame_id
    msg.scale.x = 0.01
    msg.scale.y = 0.01
    msg.scale.z = 0.01
    msg.color = ColorRGBA(*to_rgba(color))
    msg.id = idx
    msg.ns = label
    msg.pose.orientation.w = 1
    msg.pose.position.x = float(position[0])
    msg.pose.position.y = float(position[1])
    msg.pose.position.z = float(position[2])

    array_msg = MarkerArray()
    array_msg.markers.append(msg)
    pub.publish(array_msg)


def plot_arrows_rviz(
        pub, starts, directions, label, idx=0, color="g", frame_id="world", s=1
):
    msg = MarkerArray()
    for i, (start_i, dir_i) in enumerate(zip(starts, directions)):
        marker_msg = Marker()
        marker_msg.header.frame_id = frame_id
        marker_msg.header.stamp = rospy.Time.now()
        marker_msg.ns = label
        marker_msg.id = 1000 * idx + i
        marker_msg.type = Marker.ARROW
        marker_msg.action = Marker.ADD
        marker_msg.pose.orientation.w = 1
        marker_msg.scale.x = 0.005 * s
        marker_msg.scale.y = 0.01 * s
        marker_msg.scale.z = 0.03 * s
        marker_msg.color = ColorRGBA(*to_rgba(color))
        end_i = start_i + dir_i
        marker_msg.points.append(ros_numpy.msgify(Point, start_i))
        marker_msg.points.append(ros_numpy.msgify(Point, end_i))

        msg.markers.append(marker_msg)
    pub.publish(msg)


def plot_points_rviz(pub, positions, idx, label, color="g", frame_id="world", s=1):
    marker_msg = Marker()
    marker_msg.header.frame_id = frame_id
    marker_msg.header.stamp = rospy.Time.now()
    marker_msg.ns = label
    marker_msg.id = idx
    marker_msg.type = Marker.SPHERE_LIST
    marker_msg.action = Marker.ADD
    marker_msg.pose.orientation.w = 1
    marker_msg.scale.x = 0.01 * s
    marker_msg.scale.y = 0.01 * s
    marker_msg.scale.z = 0.01 * s
    marker_msg.color = ColorRGBA(*to_rgba(color))
    for position in positions:
        p = Point(x=position[0], y=position[1], z=position[2])
        marker_msg.points.append(p)

    msg = MarkerArray()
    msg.markers.append(marker_msg)
    pub.publish(msg)


def plot_lines_rviz(
        pub, positions, label, idx=0, color="g", colors=None, frame_id="world", scale=0.005
):
    marker_msg = Marker()
    marker_msg.header.frame_id = frame_id
    marker_msg.header.stamp = rospy.Time.now()
    marker_msg.ns = label
    marker_msg.id = idx
    marker_msg.type = Marker.LINE_STRIP
    marker_msg.action = Marker.ADD
    marker_msg.pose.orientation.w = 1
    marker_msg.scale.x = scale
    marker_msg.scale.y = scale
    marker_msg.scale.z = scale
    marker_msg.color = ColorRGBA(*to_rgba(color))
    for t, position in enumerate(positions):
        p = Point(x=position[0], y=position[1], z=position[2])
        marker_msg.points.append(p)
        if colors is not None:
            marker_msg.colors.append(ColorRGBA(*to_rgba(colors[t])))

    array_msg = MarkerArray()
    array_msg.markers.append(marker_msg)
    pub.publish(array_msg)
