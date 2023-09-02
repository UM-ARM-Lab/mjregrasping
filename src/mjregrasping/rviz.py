import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict

import mujoco
import numpy as np
from matplotlib.colors import to_rgba
from mujoco import mju_str2Type, mju_mat2Quat, mjtGeom, mj_id2name, mjtEq

import ros_numpy
import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from geometry_msgs.msg import Point
from mjregrasping.goal_funcs import is_valid_contact
from mjregrasping.grasping import get_eq_points
from mjregrasping.my_transforms import np_wxyz_to_xyzw
from mjregrasping.physics import Physics, get_parent_child_names
from mjregrasping.homotopy_utils import make_ring_mat
from moveit_msgs.msg import PlanningScene
from ros_numpy.point_cloud2 import merge_rgb_fields
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker


def make_clear_marker():
    clear_all_marker = MarkerArray()
    clear_all_marker.markers.append(Marker(action=Marker.DELETEALL))
    return clear_all_marker


class MjRViz:
    def __init__(self, xml_path: Path, tfw: Optional[TF2Wrapper] = None):
        self.tfw = tfw
        self.mj_xml_parser = MujocoXmlExpander(xml_path)

        self.eq_constraints_pub = rospy.Publisher("eq_constraints", MarkerArray, queue_size=10)
        self.contacts_pub = rospy.Publisher("contacts", MarkerArray, queue_size=10)
        self.contacts_pub = rospy.Publisher("contacts", MarkerArray, queue_size=10)
        self.pub = rospy.Publisher('all', MarkerArray, queue_size=10)
        self.planning_markers_pub = rospy.Publisher('planning', MarkerArray, queue_size=10)
        self.scene_pub = rospy.Publisher("scene_viz", PlanningScene, queue_size=10)

    def viz(self, phy: Physics, is_planning: bool, alpha=1):
        # 3D viz in rviz
        geom_markers_msg = MarkerArray()
        for geom_id in range(phy.m.ngeom):
            geom_bodyid = phy.m.geom_bodyid[geom_id]
            geom_name = phy.m.geom(geom_id).name
            parent_name, child_name = get_parent_child_names(geom_bodyid, phy.m)

            geom_marker_msg = Marker()
            geom_marker_msg.action = Marker.ADD
            geom_marker_msg.header.frame_id = "world"
            geom_marker_msg.ns = f"{parent_name}/{child_name}/{geom_name}"
            geom_marker_msg.id = geom_id

            geom_type = phy.m.geom_type[geom_id]
            body_pos = phy.d.xpos[geom_bodyid]
            body_xmat = phy.d.xmat[geom_bodyid]
            body_xquat = np.zeros(4)
            mju_mat2Quat(body_xquat, body_xmat)
            geom_pos = phy.d.geom_xpos[geom_id]
            geom_xmat = phy.d.geom_xmat[geom_id]
            geom_xquat = np.zeros(4)
            mju_mat2Quat(geom_xquat, geom_xmat)
            geom_size = phy.m.geom_size[geom_id]
            geom_rgba = phy.m.geom_rgba[geom_id]
            geom_meshid = phy.m.geom_dataid[geom_id]

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

                geom_markers_msg.markers.append(geom_marker_msg_ball1)
                geom_markers_msg.markers.append(geom_marker_msg_ball2)
            elif geom_type == mjtGeom.mjGEOM_SPHERE:
                geom_marker_msg.type = Marker.SPHERE
                geom_marker_msg.scale.x = geom_size[0] * 2
                geom_marker_msg.scale.y = geom_size[0] * 2
                geom_marker_msg.scale.z = geom_size[0] * 2
            elif geom_type == mjtGeom.mjGEOM_MESH:
                mesh_name = mj_id2name(
                    phy.m, mju_str2Type("mesh"), geom_meshid
                )
                # skip the phy.m prefix, e.g. val/my_mesh
                if '/' in mesh_name:
                    mesh_name = mesh_name.split("/")[1]
                geom_marker_msg.type = Marker.MESH_RESOURCE
                geom_marker_msg.mesh_use_embedded_materials = True
                mesh_file = self.mj_xml_parser.get_mesh(mesh_name)
                if mesh_file is None:
                    raise RuntimeError(f"Mesh {mesh_name} not found in XML file")
                geom_marker_msg.mesh_resource = f"package://mjregrasping/models/meshes/{mesh_file}"

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
            elif geom_type == mjtGeom.mjGEOM_PLANE:
                geom_marker_msg.type = Marker.CUBE
                geom_marker_msg.scale.x = 10
                geom_marker_msg.scale.y = 10
                geom_marker_msg.scale.z = 0.0001
            else:
                rospy.loginfo_once(f"Unsupported geom type {geom_type}")
                continue

            geom_markers_msg.markers.append(geom_marker_msg)

        if is_planning:
            self.planning_markers_pub.publish(geom_markers_msg)
        else:
            self.pub.publish(geom_markers_msg)

        for body_id in range(phy.m.nbody):
            name = mj_id2name(phy.m, mju_str2Type("body"), body_id)
            pos = phy.d.xpos[body_id]
            mat = phy.d.xmat[body_id]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(mat=mat, quat=quat)
            if self.tfw:
                self.tfw.send_transform(
                    translation=pos,
                    quaternion=np_wxyz_to_xyzw(quat),
                    parent="world",
                    child=name + "_body",
                )
        for cam_id in range(phy.m.ncam):
            mcam = phy.m.camera(cam_id)
            dcam = phy.d.camera(cam_id)
            pos = dcam.xpos
            mat = dcam.xmat
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(mat=mat, quat=quat)
            if self.tfw:
                self.tfw.send_transform(
                    translation=pos,
                    quaternion=np_wxyz_to_xyzw(quat),
                    parent="world",
                    child=mcam.name + "_cam",
                )
        for geom_id in range(phy.m.ngeom):
            name = mj_id2name(phy.m, mju_str2Type("geom"), geom_id)
            pos = phy.d.geom_xpos[geom_id]
            mat = phy.d.geom_xmat[geom_id]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(mat=mat, quat=quat)
            if self.tfw:
                self.tfw.send_transform(
                    translation=pos,
                    quaternion=np_wxyz_to_xyzw(quat),
                    parent="world",
                    child=name + "_geom",
                )
        for site_id in range(phy.m.nsite):
            name = mj_id2name(phy.m, mju_str2Type("site"), site_id)
            pos = phy.d.site_xpos[site_id]
            mat = phy.d.site_xmat[site_id]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(mat=mat, quat=quat)
            if self.tfw:
                self.tfw.send_transform(
                    translation=pos,
                    quaternion=np_wxyz_to_xyzw(quat),
                    parent="world",
                    child=name + "_site",
                )

        contact_markers = MarkerArray()
        for contact_idx, contact in enumerate(phy.d.contact):
            geom1_name = phy.m.geom(contact.geom1).name
            geom2_name = phy.m.geom(contact.geom2).name

            contact_marker = Marker()
            contact_marker.action = Marker.ADD
            contact_marker.type = Marker.SPHERE
            contact_marker.header.frame_id = "world"
            contact_marker.scale.x = 0.01
            contact_marker.scale.y = 0.01
            contact_marker.scale.z = 0.01
            contact_marker.ns = f"{geom1_name}_{geom2_name}"
            contact_marker.id = contact_idx
            contact_color = 'r' if is_valid_contact(phy, geom1_name, geom2_name) else 'orange'
            contact_marker.color = ColorRGBA(*to_rgba(contact_color))
            contact_marker.pose.orientation.w = 1
            contact_marker.pose.position.x = float(contact.pos[0])
            contact_marker.pose.position.y = float(contact.pos[1])
            contact_marker.pose.position.z = float(contact.pos[2])
            contact_markers.markers.append(contact_marker)

        clear_all_marker = make_clear_marker()
        self.contacts_pub.publish(clear_all_marker)
        self.contacts_pub.publish(contact_markers)

        eqs_markers = MarkerArray()
        for eq_constraint_idx in range(phy.m.neq):
            eq = phy.m.eq(eq_constraint_idx)
            if eq.active and eq.type == mjtEq.mjEQ_CONNECT:
                eq_marker = Marker()
                eq_marker.action = Marker.ADD
                eq_marker.type = Marker.LINE_STRIP
                eq_marker.header.frame_id = "world"
                eq_marker.scale.x = 0.005
                eq_marker.pose.orientation.w = 1
                eq_marker.color = ColorRGBA(*to_rgba("y"))
                eq_marker.color.a = 0.4
                eq_marker.ns = f"eq_{eq.name}"
                # body_xpos = phy.d.xpos[eq.obj1_id]
                # body_x_axis = body_xmat[:, :, 0]
                # xpos = body_xpos + body_x_axis * offsets[:, None]

                body1_pos, body2_pos = get_eq_points(phy, eq, eq_constraint_idx)

                eq_marker.points.append(Point(*body1_pos))
                eq_marker.points.append(Point(*body2_pos))
                eqs_markers.markers.append(eq_marker)

        self.eq_constraints_pub.publish(clear_all_marker)
        self.eq_constraints_pub.publish(eqs_markers)

    def viz_scene(self, scene_msg: PlanningScene):
        self.scene_pub.publish(scene_msg)

    def skeletons(self, skeletons: Dict):
        for k, skel in skeletons.items():
            plot_lines_rviz(self.pub, skel, k)


def plot_spheres_rviz(pub, positions, colors, radius, frame_id="world", idx=0, label=""):
    msg = Marker()
    msg.action = Marker.ADD
    msg.type = Marker.SPHERE_LIST
    msg.header.frame_id = frame_id
    msg.scale.x = radius * 2
    msg.scale.y = radius * 2
    msg.scale.z = radius * 2
    msg.id = idx
    msg.ns = label
    msg.pose.orientation.w = 1
    for p, c in zip(positions, colors):
        msg.points.append(Point(*p))
        msg.colors.append(ColorRGBA(*to_rgba(c)))

    array_msg = MarkerArray()
    array_msg.markers.append(msg)
    pub.publish(array_msg)


def plot_sphere_rviz(pub, position, radius, frame_id="world", color="m", idx=0, label=""):
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


def plot_arrows_rviz(pub, starts, directions, label, idx=0, color="g", frame_id="world", s=1):
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


def plot_lines_rviz(pub, positions, label, idx=0, color="g", colors=None, frame_id="world", scale=0.005):
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


def plot_ring_rviz(pub, ring_position, ring_z_axis, radius, idx=0):
    ring_msg = Marker()
    ring_msg.header.frame_id = "world"
    ring_msg.ns = "ring"
    ring_msg.id = idx
    ring_msg.type = Marker.LINE_STRIP
    ring_msg.action = Marker.ADD
    ring_msg.pose.orientation.w = 1.0
    ring_msg.scale.x = radius * 0.2
    ring_msg.color.a = 1.0
    ring_msg.color.g = 1.0

    delta_angle = 0.1
    angles = np.arange(0, 2 * np.pi, delta_angle)
    zeros = np.zeros_like(angles)
    ones = np.ones_like(angles)
    x = np.stack([radius * np.cos(angles), radius * np.sin(angles), zeros, ones], -1)
    ring_mat = make_ring_mat(ring_position, ring_z_axis)
    x = (x @ ring_mat.T)[:, :3]

    for x_i in x:
        ring_msg.points.append(Point(x_i[0], x_i[1], x_i[2]))
    ring_msg.points.append(ring_msg.points[0])
    markers_msg = MarkerArray()
    markers_msg.markers.append(ring_msg)

    pub.publish(markers_msg)


class MujocoXmlExpander:

    def __init__(self, xml_path: Path):
        self.xml_path = xml_path
        self.tree = ET.parse(self.xml_path)
        self.root = self.tree.getroot()
        for include in self.root.findall('include'):
            file = include.attrib['file']
            with open(f'models/{file}', 'r') as include_file:
                include_root = ET.fromstring(include_file.read())
                self.root.extend(include_root)

        # remove include elements now that they've been "included"
        for include in self.root.findall('include'):
            self.root.remove(include)

    def get_mesh(self, mesh_name):
        for asset in self.root.findall("asset"):
            for mesh in asset.findall("mesh"):
                name = mesh.attrib['name']
                file = mesh.attrib['file']
                if name == mesh_name:
                    return file
        return None

    # iterate over all worldbody elements
    # and all body elements within them recursively
    def get_e(self, tag: str, name: str):
        for e in self.root.iter(tag):
            if "name" in e.attrib and e.attrib["name"] == name:
                return e

    def get_vec(self, e, k: str):
        x_str = e.attrib[k]
        x = [float(x) for x in x_str.split()]
        return x

    def set_vec(self, e, x, k: str):
        x_str = " ".join([str(x) for x in x])
        e.attrib[k] = x_str

    def set_vec_i(self, e, k: str, i: int, x_i: float):
        x = self.get_vec(e, k)
        x[i] = x_i
        self.set_vec(e, x, k)

    def save_tmp(self, tmp_filename='models/tmp.xml'):
        self.tree.write(tmp_filename)
        return Path(tmp_filename)


def pc_np_to_pc_msg(pc, names, frame_id):
    """

    Args:
        pc: [M, N] array where M is probably either 3 or 6
        names: strings of comma separated names of the fields in pc, e.g. 'x,y,z' or 'x,y,z,r,g,b'
        frame_id: string

    Returns:
        PointCloud2 message

    """
    pc_rec = np.rec.fromarrays(pc, names=names)
    if 'r' in names:
        pc_rec = merge_rgb_fields(pc_rec)
    pc_msg = ros_numpy.msgify(PointCloud2, pc_rec, stamp=rospy.Time.now(), frame_id=frame_id)
    return pc_msg
