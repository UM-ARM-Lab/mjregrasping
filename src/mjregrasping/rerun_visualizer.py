import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict

import mujoco
import numpy as np
import rerun as rr
from matplotlib.colors import to_rgba
from mujoco import mjtSensor, mjtGeom, mj_id2name
from mujoco._structs import _MjDataGeomViews, _MjModelGeomViews
from trimesh.creation import box, cylinder

from mjregrasping.grasping import get_eq_points
from mjregrasping.my_transforms import np_wxyz_to_xyzw
from mjregrasping.physics import Physics, get_total_contact_force, get_parent_child_names
from mjregrasping.rviz import MujocoXmlMeshParser

logger = logging.getLogger(f'rosout.{__name__}')


def init():
    rr.set_time_seconds('sim_time', 0.0)
    rr.log_view_coordinates("world", up="+Z", timeless=True)
    rr.log_arrow('world_x', [0, 0, 0], [1, 0, 0], color=(255, 0, 0), width_scale=0.02)
    rr.log_arrow('world_y', [0, 0, 0], [0, 1, 0], color=(0, 255, 0), width_scale=0.02)
    rr.log_arrow('world_z', [0, 0, 0], [0, 0, 1], color=(0, 0, 255), width_scale=0.02)


class MjReRun:

    def __init__(self, xml_path):
        self.mj_xml_parser = MujocoXmlMeshParser(xml_path)
        init()

    def viz(self, phy: Physics, is_planning=False, detailed=False):
        entity_prefix = 'planning/' if is_planning else ''
        rr.set_time_seconds('sim_time', phy.d.time)

        # 2D viz in rerun
        for sensor_idx in range(phy.m.nsensor):
            sensor = phy.m.sensor(sensor_idx)
            if sensor.type in [mjtSensor.mjSENS_TORQUE, mjtSensor.mjSENS_FORCE]:
                rr.log_scalar(f'sensor/{sensor.name}', float(phy.d.sensordata[sensor.adr]))
        for joint_idx in range(phy.m.njnt):
            joint = phy.m.joint(joint_idx)
            if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                rr.log_scalar(f'qpos/{joint.name}', float(phy.d.qpos[joint.qposadr]))
        for act_i in range(phy.m.na):
            act = phy.m.actuator(act_i)
            rr.log_scalar(f'ctrl/{act.name}', float(phy.d.ctrl[act.id]))

        rr.log_scalar(f'contact/num_contacts', len(phy.d.contact))

        self.viz_bodies(phy.m, phy.d, entity_prefix, detailed)
        self.viz_sites(phy, entity_prefix)
        self.viz_contacts(phy, entity_prefix)
        self.viz_eqs(phy, entity_prefix)

    def viz_sites(self, phy: Physics, entity_prefix):
        for site_id in range(phy.m.nsite):
            name = phy.m.site(site_id).name
            pos = phy.d.site_xpos[site_id]
            rr.log_point(entity_path=f'sites/{name}', position=pos)

    def viz_bodies(self, m: mujoco.MjModel, d: mujoco.MjData, entity_prefix, detailed=False):
        """
        Rerun, or possibly my code, seems to have some serious problems with efficiency when logging meshes,
        so this method is very hacked at the moment.
        """
        for geom_id in range(m.ngeom):
            d_geom = d.geom(geom_id)
            m_geom = m.geom(geom_id)
            geom_type = m_geom.type
            geom_bodyid = m_geom.bodyid
            parent_name, child_name = get_parent_child_names(geom_bodyid, m)
            entity_name = f"{entity_prefix}{parent_name}/{child_name}"
            d_body = d.body(geom_bodyid)
            m_body = m.body(geom_bodyid)
            d_parent_body = d.body(m_body.parentid)
            m_parent_body = m.body(m_body.parentid)

            if geom_type == mjtGeom.mjGEOM_BOX:
                if detailed:
                    log_box_from_geom(entity_name, m_geom, d_geom)
                else:
                    log_bbox_from_geom(entity_name, m_geom, d_geom)
            elif geom_type == mjtGeom.mjGEOM_CYLINDER:
                if detailed:
                    log_cylinder(entity_name, m_geom, d_geom)
            elif geom_type == mjtGeom.mjGEOM_CAPSULE:
                log_capsule(entity_name, m_geom, d_geom)
            elif geom_type == mjtGeom.mjGEOM_SPHERE:
                log_sphere(entity_name, m_geom, d_geom)
            elif geom_type == mjtGeom.mjGEOM_MESH:
                if detailed:
                    mesh_file_contents = self.get_mesh_file_contents(m_geom, m)
                    log_mesh_body(entity_name, d_body, mesh_file_contents)
                else:
                    # We use body pos/quat here under the assumption that in the XML, the <geom type="mesh" ... />
                    # has NO POS OR QUAT, but instead that info goes in the <body> tag
                    if m_parent_body.name != 'val_base':
                        log_robot_skeleton(entity_name, m_body, d_body, d_parent_body, color=[0.8, 0.8, 0.8, 1.0])
            else:
                logger.debug(f"Unsupported geom type {geom_type}")
                continue

    @lru_cache
    def get_mesh_file_contents(self, geom, m):
        mesh_name = mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, geom.dataid)
        # skip the model prefix, e.g. val/my_mesh
        if '/' in mesh_name:
            mesh_name = mesh_name.split("/")[1]
        mesh_file = Path(self.mj_xml_parser.get_mesh(mesh_name))
        mesh_file = mesh_file.stem + ".glb"
        if mesh_file is None:
            raise RuntimeError(f"Mesh {mesh_name} not found in XML file")
        mesh_file = Path.home() / "mjregrasping_ws/src/mjregrasping/models/meshes" / mesh_file
        if not mesh_file.exists():
            raise RuntimeError(f"Mesh {mesh_file} not found on disk")
        with open(mesh_file, 'rb') as f:
            mesh_file_contents = f.read()
        return mesh_file_contents

    def viz_contacts(self, phy: Physics, entity_prefix):
        rr.log_cleared('contacts', recursive=True)
        positions = []
        radii = []
        colors = []
        for contact_idx, contact in enumerate(phy.d.contact):
            positions.append(contact.pos)
            radii.append(0.01)
            colors.append((255, 0, 0, 128))

        rr.log_points(entity_path=f"{entity_prefix}contacts",
                      positions=positions,
                      colors=colors,
                      radii=radii)

        total_contact_force = get_total_contact_force(phy)
        rr.log_scalar('total_contact_force', total_contact_force)

    def viz_eqs(self, phy: Physics, entity_prefix):
        rr.log_cleared('eqs', recursive=True)
        for eq_constraint_idx in range(phy.m.neq):
            eq = phy.m.eq(eq_constraint_idx)
            if eq.active and eq.type == mujoco.mjtEq.mjEQ_CONNECT:
                color = list(to_rgba("y"))
                color[-1] = 0.4
                points = get_eq_points(phy, eq, eq_constraint_idx)
                entity_path = "/".join(filter(None, ["eqs", entity_prefix, eq.name]))
                rr.log_line_strip(entity_path, points, color=color)

    def sdf(self, sdf):
        points = []
        colors = []
        for x_i in range(0, sdf.GetNumXCells(), 1):
            for y_i in range(0, sdf.GetNumYCells(), 1):
                for z_i in [0]:  # range(0, sdf.GetNumZCells(), 1):
                    origin = sdf.GetOriginTransform().translation()
                    p = origin + np.array([x_i, y_i, z_i]) * sdf.GetResolution()
                    points.append(p)
                    sdf_value = sdf.GetValueByIndex(x_i, y_i, z_i)[0]
                    colors.append([1, 0, 0, 1.0] if sdf_value < 0 else [0, 1, 0, 1.0])
        rr.log_points(f'sdf', positions=points, colors=colors, radii=sdf.GetResolution() / 2)


def make_entity_path(*names):
    """ joins names with slashes but ignores empty names """
    return '/'.join([name for name in names if name])


def get_transform(data: _MjDataGeomViews):
    xmat = data.xmat.reshape(3, 3)
    transform = np.eye(4)
    transform[0:3, 0:3] = xmat
    transform[0:3, 3] = data.xpos
    return transform


def log_plane(body_name, model: _MjModelGeomViews, data: _MjDataGeomViews):
    transform = get_transform(data)
    entity_path = make_entity_path(body_name, model.name)
    mesh = box(np.array([model.size[0], model.size[1], 0.001]), transform)
    rr.log_mesh(entity_path=entity_path,
                positions=mesh.vertices,
                indices=mesh.faces,
                albedo_factor=model.rgba)


def log_capsule(body_name, model: _MjModelGeomViews, data: _MjDataGeomViews):
    entity_path = make_entity_path(body_name, model.name)
    xmat = data.xmat.reshape(3, 3)
    start = data.xpos - model.size[1] * xmat[:, 2]
    end = data.xpos + model.size[1] * xmat[:, 2]
    rr.log_line_strip(entity_path=entity_path + '/line',
                      positions=np.stack([start, end]),
                      stroke_width=model.size[0] * 2,
                      color=tuple(model.rgba))
    rr.log_point(entity_path=entity_path + '/start',
                 position=start,
                 radius=model.size[0],
                 color=tuple(model.rgba))
    rr.log_point(entity_path=entity_path + '/end',
                 position=end,
                 radius=model.size[0],
                 color=tuple(model.rgba))


def log_cylinder(body_name, model: _MjModelGeomViews, data: _MjDataGeomViews):
    transform = get_transform(data)
    entity_path = make_entity_path(body_name, model.name)
    mesh = cylinder(radius=model.size[0], height=2 * model.size[1], transform=transform, sections=16)
    rr.log_mesh(entity_path=entity_path,
                positions=mesh.vertices,
                indices=mesh.faces,
                albedo_factor=model.rgba)


def log_line(body_name, model: _MjModelGeomViews, data: _MjDataGeomViews):
    entity_path = make_entity_path(body_name, model.name)
    xmat = data.xmat.reshape(3, 3)
    start = data.xpos - model.size[1] * xmat[:, 2]
    end = data.xpos + model.size[1] * xmat[:, 2]
    rr.log_line_strip(entity_path=entity_path,
                      positions=np.stack([start, end]),
                      stroke_width=model.size[0],
                      color=tuple(model.rgba))


def log_sphere(body_name, model: _MjModelGeomViews, data: _MjDataGeomViews):
    entity_path = make_entity_path(body_name, model.name)
    rr.log_point(entity_path=entity_path,
                 position=data.xpos,
                 radius=model.size[0],
                 color=tuple(model.rgba))


def log_robot_skeleton(body_name, model, data, parent_data, **kwargs):
    entity_path = make_entity_path(body_name, model.name)
    rr.log_point(entity_path=entity_path,
                 position=data.xpos,
                 **kwargs)
    rr.log_line_strip(entity_path=entity_path,
                      positions=[data.xpos, parent_data.xpos],
                      **kwargs)


def log_mesh_body(entity_path, data, mesh_file_contents):
    transform = get_transform(data)
    rr.log_mesh_file(entity_path=entity_path,
                     mesh_format=rr.MeshFormat.GLB,
                     mesh_file=mesh_file_contents,
                     transform=transform[:3, :])


def log_box_from_geom(body_name, model: _MjModelGeomViews, data: _MjDataGeomViews):
    transform = get_transform(data)
    log_box(make_entity_path(body_name, model.name), model.size * 2, transform, model.rgba)


def log_bbox_from_geom(body_name, model: _MjModelGeomViews, data: _MjDataGeomViews):
    xquat = np.zeros(4)
    mujoco.mju_mat2Quat(xquat, data.xmat)
    xquat = np_wxyz_to_xyzw(xquat)
    entity_path = make_entity_path(body_name, model.name)
    rr.log_obb(entity_path=entity_path,
               half_size=model.size,
               position=data.xpos,
               rotation_q=xquat,
               color=tuple(model.rgba))


def log_box(entity_path, size, transform, color):
    mesh = box(size, transform)
    rr.log_mesh(entity_path=entity_path,
                positions=mesh.vertices,
                indices=mesh.faces,
                albedo_factor=color)


def log_line_with_std(entity_path, x, *args, **kwargs):
    """

    Args:
        entity_path: Entity apth
        x: [N,] array, where N are different samples
    Returns:

    """
    mean = np.mean(x)
    std = np.std(x)
    low = mean - std
    high = mean + std
    rr.log_scalar(f'{entity_path}/μ', mean, *args, **kwargs)
    rr.log_scalar(f'{entity_path}/μ-σ', low, *args, **kwargs)
    rr.log_scalar(f'{entity_path}/μ+σ', high, *args, **kwargs)


def log_skeletons(skeletons: Dict[str, np.ndarray], **kwargs):
    rr.log_cleared(f'skeleton', recursive=True)
    for name, skeleton in skeletons.items():
        rr.log_line_strip(f'skeleton/{name}', skeleton, **kwargs)
