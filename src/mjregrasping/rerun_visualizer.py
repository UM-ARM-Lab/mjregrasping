import logging
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
from typing import Dict

import mujoco
import numpy as np
import rerun as rr
from matplotlib.colors import to_rgba
from mujoco import mjtGeom, mj_id2name
from trimesh.creation import box, cylinder, capsule, icosphere

from mjregrasping.grasping import get_eq_points
from mjregrasping.mjxml_expander import MujocoXmlExpander
from transformations import quaternion_from_matrix
from mjregrasping.physics import Physics, get_total_contact_force, get_parent_child_names

logger = logging.getLogger(f'rosout.{__name__}')


def init():
    rr.set_time_seconds('sim_time', 0.0)
    rr.log('world', rr.Transform3D())


class MjReRun:

    def __init__(self, xml_path):
        self.mj_xml_parser = MujocoXmlExpander(xml_path)
        init()

        # We only need to log things once, and then we can just update the transform
        self.entity_cache = {}

    def viz(self, phy: Physics, is_planning=False):
        entity_prefix = 'planning/' if is_planning else ''
        rr.set_time_seconds('sim_time', phy.d.time)

        self.viz_bodies(phy.m, phy.d, entity_prefix)
        self.viz_sites(phy, entity_prefix)
        self.viz_contacts(phy, entity_prefix)
        self.viz_eqs(phy, entity_prefix)

    def viz_sites(self, phy: Physics, entity_prefix):
        for site_id in range(phy.m.nsite):
            name = phy.m.site(site_id).name
            pos = phy.d.site_xpos[site_id]
            rr.log(f'sites/{name}', rr.Points3D(pos))

    def viz_bodies(self, m: mujoco.MjModel, d: mujoco.MjData, entity_prefix):
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
            entity_name = f"{entity_prefix}{parent_name}/{child_name}/{m_geom.name}"
            d_body = d.body(geom_bodyid)

            if geom_type == mjtGeom.mjGEOM_BOX:
                size = m_geom.size
                mesh = box(2 * size)
                rr.log(entity_name, rr.Mesh3D(vertex_positions=mesh.vertices, triangle_indices=mesh.faces, vertex_colors=m_geom.rgba))
                rr.log(entity_name, rr.Transform3D(rr.TranslationAndMat3x3(d_geom.xpos, d_geom.xmat.reshape(3, 3))))
            elif geom_type == mjtGeom.mjGEOM_CYLINDER:
                mesh = cylinder(radius=m_geom.size[0], height=2 * m_geom.size[1], sections=16)
                rr.log(entity_name, rr.Mesh3D(vertex_positions=mesh.vertices, triangle_indices=mesh.faces, vertex_colors=m_geom.rgba))
                rr.log(entity_name, rr.Transform3D(rr.TranslationAndMat3x3(d_geom.xpos, d_geom.xmat.reshape(3, 3))))
            elif geom_type == mjtGeom.mjGEOM_CAPSULE:
                mesh = capsule(radius=m_geom.size[0], height=2 * m_geom.size[1], count=[8, 16])
                rr.log(entity_name, rr.Mesh3D(vertex_positions=mesh.vertices, triangle_indices=mesh.faces, vertex_colors=m_geom.rgba))
                rr.log(entity_name, rr.Transform3D(rr.TranslationAndMat3x3(d_geom.xpos, d_geom.xmat.reshape(3, 3))))
            elif geom_type == mjtGeom.mjGEOM_SPHERE:
                mesh = icosphere(subdivisions=4, radius=m_geom.size[0])
                rr.log(entity_name, rr.Mesh3D(vertex_positions=mesh.vertices, triangle_indices=mesh.faces, vertex_colors=m_geom.rgba))
                rr.log(entity_name, rr.Transform3D(rr.TranslationAndMat3x3(d_geom.xpos, d_geom.xmat.reshape(3, 3))))
            elif geom_type == mjtGeom.mjGEOM_PLANE:
                mesh = box(np.array([m_geom.size[0], m_geom.size[1], 0.001]))
                rr.log(entity_name, rr.Mesh3D(vertex_positions=mesh.vertices, triangle_indices=mesh.faces, vertex_colors=m_geom.rgba))
                rr.log(entity_name, rr.Transform3D(rr.TranslationAndMat3x3(d_geom.xpos, d_geom.xmat.reshape(3, 3))))
            elif geom_type == mjtGeom.mjGEOM_MESH:
                mesh_path = self.get_mesh_path(m_geom, m)
                rr.log(entity_name, rr.Asset3D(path=mesh_path))
                # for meshes, we need to use the body not the geom
                rr.log(entity_name, rr.Transform3D(rr.TranslationAndMat3x3(d_body.xpos, d_body.xmat.reshape(3, 3))))
            else:
                logger.debug(f"Unsupported geom type {geom_type}")
                continue

    def get_mesh_path(self, geom, m):
        mesh_name = mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, geom.dataid)
        # skip the model prefix, e.g. val/my_mesh
        if '/' in mesh_name:
            mesh_name = mesh_name.split("/")[1]
        mesh_path = Path(self.mj_xml_parser.get_mesh(mesh_name))
        if mesh_path is None:
            raise RuntimeError(f"Mesh {mesh_name} not found in XML file")
        abs_mesh_path = ROOT_DIR / "models" / "meshes" / mesh_path
        if not abs_mesh_path.exists():
            raise RuntimeError(f"Mesh {mesh_path} not found on disk")
        return abs_mesh_path

    def viz_contacts(self, phy: Physics, entity_prefix):
        rr.log('contacts', rr.Clear(recursive=True))
        positions = []
        radii = []
        colors = []
        for contact_idx, contact in enumerate(phy.d.contact):
            positions.append(contact.pos)
            radii.append(0.01)
            colors.append((255, 0, 0, 128))

        rr.log(f"{entity_prefix}contacts", rr.Points3D(positions, colors=colors, radii=radii))

        total_contact_force = get_total_contact_force(phy)
        rr.log('total_contact_force', rr.Scalar(total_contact_force))

    def viz_eqs(self, phy: Physics, entity_prefix):
        rr.log('eqs', rr.Clear(recursive=True))
        for eq_constraint_idx in range(phy.m.neq):
            eq = phy.m.eq(eq_constraint_idx)
            if phy.d.eq_active[eq.id] and eq.type in [mujoco.mjtEq.mjEQ_CONNECT, mujoco.mjtEq.mjEQ_WELD]:
                color = list(to_rgba("y"))
                color[-1] = 0.4
                points = get_eq_points(phy, eq)
                entity_path = "eqs"
                if entity_prefix is not None and entity_prefix != "":
                    entity_path += "/" + entity_path
                if eq.name != "":
                    entity_path += "/" + eq.name
                rr.log(entity_path, rr.LineStrips3D(points, colors=color))


def make_entity_path(*names):
    """ joins names with slashes but ignores empty names """
    return '/'.join([name for name in names if name])


def log_skeletons(skeletons: Dict[str, np.ndarray], **kwargs):
    rr.log(f'skeleton', rr.Clear(recursive=True))
    for name, skeleton in skeletons.items():
        rr.log(f'skeleton/{name}', rr.LineStrips3D(skeleton, **kwargs))
