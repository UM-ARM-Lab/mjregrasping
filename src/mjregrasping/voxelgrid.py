import mujoco
import numpy as np

import rospy
from mjregrasping.body_with_children import Objects
from mjregrasping.physics import Physics
from visualization_msgs.msg import MarkerArray


def point_to_idx(points, origin_point, res):
    # round helps with stupid numerics issues
    return np.round((points - origin_point) / res).astype(int)


def idx_to_point_from_origin_point(i, res, origin_point):
    return origin_point + i.astype(float) * res - res / 2


def extent_to_env_size(extent):
    xmin, xmax, ymin, ymax, zmin, zmax = extent
    env_x_m = abs(xmax - xmin)
    env_y_m = abs(ymax - ymin)
    env_z_m = abs(zmax - zmin)
    return env_x_m, env_y_m, env_z_m


def extent_to_env_shape(extent, res):
    extent = np.array(extent).astype(np.float32)
    res = np.float32(res)
    env_x_m, env_y_m, env_z_m = extent_to_env_size(extent)
    x_shape = int(env_x_m / res)
    y_shape = int(env_y_m / res)
    z_shape = int(env_z_m / res)
    return np.array([x_shape, y_shape, z_shape])


def set_cc_sphere_pos(phy, xyz):
    qposadr = int(phy.m.joint("cc_sphere").qposadr)
    phy.d.qpos[qposadr: qposadr + 3] = xyz


class VoxelGrid:

    def __init__(self, phy: Physics, res, extends_2d):
        pub = rospy.Publisher("dfield", MarkerArray, queue_size=10)
        self.objects = Objects(phy.m)
        self.res = res
        self.extents_2d = extends_2d
        self.extents_flat = self.extents_2d.reshape(-1)
        self.shape = extent_to_env_shape(self.extents_flat, self.res)
        xmin = self.extents_2d[0, 0]
        ymin = self.extents_2d[1, 0]
        zmin = self.extents_2d[2, 0]
        self.origin_point = np.array([xmin, ymin, zmin]) + self.res / 2  # center  of the voxel [0,0,0]

        self.vg = np.zeros(self.shape, dtype=np.float32)
        for x_i, y_i, z_i in list(np.ndindex(*self.shape)):
            xyz = idx_to_point_from_origin_point(np.array([x_i, y_i, z_i]), self.res, self.origin_point)
            set_cc_sphere_pos(phy, xyz)

            mujoco.mj_step1(phy.m, phy.d)  # call step to update collisions
            for c in phy.d.contact:
                geom1_name = phy.m.geom(c.geom1).name
                geom2_name = phy.m.geom(c.geom2).name
                obs = geom1_name in self.objects.obstacle.geom_names or geom2_name in self.objects.obstacle.geom_names
                cc = geom1_name == 'cc_sphere' or geom2_name == 'cc_sphere'
                if c.dist < 0 and cc and obs:
                    self.vg[x_i, y_i, z_i] = 1
                    # print(f"Contact at {xyz} between {geom1_name} and {geom2_name}, {xyz}")
                    # marker = Marker()
                    # marker.id = idx
                    # idx += 1
                    # marker.header.frame_id = "world"
                    # marker.ns = 'vg'
                    # marker.type = marker.SPHERE
                    # marker.action = marker.ADD
                    # marker.scale.x = res / 2
                    # marker.scale.y = res / 2
                    # marker.scale.z = res / 2
                    # marker.color.a = 0.5
                    # marker.color.r = 1.0
                    # marker.pose.orientation.w = 1.0
                    # marker.pose.position.x = xyz[0]
                    # marker.pose.position.y = xyz[1]
                    # marker.pose.position.z = xyz[2]
                    # markers = MarkerArray()
                    # markers.markers.append(marker)
                    # pub.publish(markers)
                    # rospy.sleep(0.005)

                    break
        set_cc_sphere_pos(phy, np.array([10, 0, 0]))
