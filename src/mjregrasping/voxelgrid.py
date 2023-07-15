import mujoco
import numpy as np
import pysdf_tools

from mjregrasping.mujoco_object import MjObject
from mjregrasping.physics import Physics


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


def make_vg(phy: Physics, res, extends_2d, origin_point, obstacle: MjObject):
    phy = phy.copy_data()  # make a copy, so we don't mess up the state for the caller
    obstacle = obstacle
    res = res
    extents_2d = extends_2d
    extents_flat = extents_2d.reshape(-1)
    shape = extent_to_env_shape(extents_flat, res)

    origin_transform = pysdf_tools.Isometry3d([
        [1.0, 0.0, 0.0, origin_point[0]],
        [0.0, 1.0, 0.0, origin_point[1]],
        [0.0, 0.0, 1.0, origin_point[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

    oob_value = pysdf_tools.COLLISION_CELL(-10000)
    occupied_value = pysdf_tools.COLLISION_CELL(1)
    grid = pysdf_tools.CollisionMapGrid(origin_transform, 'world', res, *shape, oob_value)
    points = []
    for x_i, y_i, z_i in list(np.ndindex(*shape)):
        xyz = idx_to_point_from_origin_point(np.array([x_i, y_i, z_i]), res, origin_point)
        set_cc_sphere_pos(phy, xyz)

        mujoco.mj_step1(phy.m, phy.d)  # call step to update collisions
        for c in phy.d.contact:
            geom1_name = phy.m.geom(c.geom1).name
            geom2_name = phy.m.geom(c.geom2).name
            is_obs = geom1_name in obstacle.geom_names or geom2_name in obstacle.geom_names
            cc = geom1_name == 'cc_sphere' or geom2_name == 'cc_sphere'
            if c.dist < 0 and cc and is_obs:
                grid.SetValue(x_i, y_i, z_i, occupied_value)
                points.append(xyz)
                break

    return grid, points


def get_points_and_values(sdf, indices):
    points = []
    values = []
    for x_i, y_i, z_i in indices:
        origin = sdf.GetOriginTransform().translation()
        p = origin + np.array([x_i, y_i, z_i]) * sdf.GetResolution()
        points.append(p)
        sdf_value = sdf.GetValueByIndex(x_i, y_i, z_i)[0]
        values.append(sdf_value)
    return points, values
