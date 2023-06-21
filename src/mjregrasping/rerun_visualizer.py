import logging
from pathlib import Path

import mujoco
import numpy as np
import rerun as rr
from mujoco import mjtSensor, mjtGeom, mj_id2name
from mujoco._structs import _MjDataGeomViews, _MjModelGeomViews
from trimesh.creation import box, cylinder, capsule

from mjregrasping.physics import Physics
from mjregrasping.rviz import get_parent_child_names, MujocoXmlMeshParser

logger = logging.getLogger(f'rosout.{__name__}')


class MjReRun:

    def __init__(self, xml_path):
        self.mj_xml_parser = MujocoXmlMeshParser(xml_path)
        rr.set_time_seconds('sim_time', 0.0)
        rr.log_view_coordinates("world", up="+Z", timeless=True)
        rr.log_arrow('world_x', [0, 0, 0], [1, 0, 0], color=(255, 0, 0), width_scale=0.02)
        rr.log_arrow('world_y', [0, 0, 0], [0, 1, 0], color=(0, 255, 0), width_scale=0.02)
        rr.log_arrow('world_z', [0, 0, 0], [0, 0, 1], color=(0, 0, 255), width_scale=0.02)

    def viz(self, phy: Physics):
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

        rr.log_scalar(f'contact/num_contacts', len(phy.d.contact))

        # self.viz_bodies(phy.m, phy.d)

    def viz_bodies(self, m: mujoco.MjModel, d: mujoco.MjData):
        for geom_id in range(m.ngeom):
            geom = m.geom(geom_id)
            geom_type = geom.type
            geom_bodyid = geom.bodyid
            parent_name, child_name = get_parent_child_names(geom_bodyid, m)
            entity_name = f"{parent_name}/{child_name}"

            if geom_type == mjtGeom.mjGEOM_BOX:
                log_box_from_geom(entity_name, m.geom(geom_id), d.geom(geom_id))
            elif geom_type == mjtGeom.mjGEOM_CYLINDER:
                log_cylinder(entity_name, m.geom(geom_id), d.geom(geom_id))
            elif geom_type == mjtGeom.mjGEOM_CAPSULE:
                log_capsule(entity_name, m.geom(geom_id), d.geom(geom_id))
                pass
            elif geom_type == mjtGeom.mjGEOM_SPHERE:
                log_sphere(entity_name, m.geom(geom_id), d.geom(geom_id))
            elif geom_type == mjtGeom.mjGEOM_MESH:
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
                # We use body pos/quat here under the assumption that in the XML, the <geom type="mesh" ... />
                # has NO POS OR QUAT, but instead that info goes in the <body> tag
                log_mesh(entity_name, m.body(geom_bodyid), d.body(geom_bodyid), mesh_file)
            else:
                logger.error(f"Unsupported geom type {geom_type}")
                continue


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
    transform = get_transform(data)
    mesh = capsule(radius=model.size[0], height=2 * model.size[1], transform=transform, count=[6, 6])
    rr.log_mesh(entity_path=entity_path,
                positions=mesh.vertices,
                indices=mesh.faces,
                albedo_factor=model.rgba)


def log_cylinder(body_name, model: _MjModelGeomViews, data: _MjDataGeomViews):
    transform = get_transform(data)
    entity_path = make_entity_path(body_name, model.name)
    mesh = cylinder(radius=model.size[0], height=2 * model.size[1], transform=transform, sections=16)
    rr.log_mesh(entity_path=entity_path,
                positions=mesh.vertices,
                indices=mesh.faces,
                albedo_factor=model.rgba)


def log_sphere(body_name, model: _MjModelGeomViews, data: _MjDataGeomViews):
    entity_path = make_entity_path(body_name, model.name)
    rr.log_point(entity_path=entity_path,
                 position=data.xpos,
                 radius=model.size[0],
                 color=tuple(model.rgba))


def log_mesh(body_name, model, data, mesh_file):
    transform = get_transform(data)
    entity_path = make_entity_path(body_name, model.name)
    if entity_path not in prim_mesh_cache:
        with open(mesh_file, 'rb') as f:
            mesh_file_contents = f.read()
        prim_mesh_cache[entity_path] = mesh_file_contents
    else:
        mesh_file_contents = prim_mesh_cache[entity_path]
    rr.log_mesh_file(entity_path=entity_path,
                     mesh_format=rr.MeshFormat.GLB,
                     mesh_file=mesh_file_contents,
                     transform=transform[:3, :])


def log_box_from_geom(body_name, model: _MjModelGeomViews, data: _MjDataGeomViews):
    transform = get_transform(data)
    log_box(make_entity_path(body_name, model.name), model.size * 2, transform, model.rgba)


def log_box(entity_path, size, transform, color):
    mesh = box(size, transform)
    rr.log_mesh(entity_path=entity_path,
                positions=mesh.vertices,
                indices=mesh.faces,
                albedo_factor=color)
