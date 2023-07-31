import numpy as np
from mujoco import mju_mat2Quat, mjtGeom

import rospy
from geometry_msgs.msg import Pose
from mjregrasping.my_transforms import np_wxyz_to_xyzw
from mjregrasping.physics import Physics, get_full_q, get_parent_child_names
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive


def make_planning_scene(phy: Physics):
    scene_msg = PlanningScene()
    scene_msg.is_diff = True
    scene_msg.robot_state.is_diff = True

    q = get_full_q(phy)
    scene_msg.robot_state.joint_state.name = phy.o.robot.joint_names
    scene_msg.robot_state.joint_state.position = q.tolist()

    # Collision objects
    for geom_id in phy.o.obstacle.geom_indices:
        geom_bodyid = phy.m.geom_bodyid[geom_id]
        geom_name = phy.m.geom(geom_id).name
        parent_name, child_name = get_parent_child_names(geom_bodyid, phy.m)

        co = CollisionObject()
        co.operation = CollisionObject.ADD
        co.header.frame_id = "robot_root"
        co.id = f"{parent_name}/{child_name}/{geom_name}"

        geom_type = phy.m.geom_type[geom_id]
        geom_pos = phy.d.geom_xpos[geom_id]
        geom_xmat = phy.d.geom_xmat[geom_id]
        geom_xquat = np.zeros(4)
        mju_mat2Quat(geom_xquat, geom_xmat)
        geom_xquat = np_wxyz_to_xyzw(geom_xquat)
        geom_size = phy.m.geom_size[geom_id]

        co.pose.position.x = geom_pos[0]
        co.pose.position.y = geom_pos[1]
        co.pose.position.z = geom_pos[2]
        co.pose.orientation.w = geom_xquat[0]
        co.pose.orientation.x = geom_xquat[1]
        co.pose.orientation.y = geom_xquat[2]
        co.pose.orientation.z = geom_xquat[3]

        prim = SolidPrimitive()
        prim_pose = Pose()
        prim_pose.orientation.w = 1.0

        if geom_type == mjtGeom.mjGEOM_BOX:
            prim.type = SolidPrimitive.BOX
            prim.dimensions = [geom_size[0] * 2, geom_size[1] * 2, geom_size[2] * 2]
            co.primitives.append(prim)
            co.primitive_poses.append(prim_pose)
        else:
            rospy.loginfo_once(f"Unsupported geom type {geom_type}")
            continue

        scene_msg.world.collision_objects.append(co)

    return scene_msg
