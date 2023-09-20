from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
from mujoco import mjtGeom, mj_id2name
from vedo import Plotter, Video, Box, Cylinder, Sphere, load, Points

from mjregrasping.mujoco_object import MjObject
from mjregrasping.physics import Physics
from mjregrasping.rviz import MujocoXmlExpander
from mjregrasping.trials import load_phy_and_skeletons

COLORS = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w']


class MjVedo:

    def __init__(self, xml_path, fps=60):
        self.mj_xml_parser = MujocoXmlExpander(xml_path)
        self.fps = fps

        self.plotter = Plotter(title="mjvedo", axes=0, interactive=False)

        self.mesh_cache = {}
        # maps from mujoco geom IDs to Vedo actors
        self.actor_map = {}

    def record(self, filename, anim_func: Callable, num_frames: int):
        video = Video(filename, fps=self.fps, backend="ffmpeg")

        for t in range(num_frames):
            done = anim_func(t, self.plotter)
            self.plotter.show(interactive=False, resetcam=False)
            video.add_frame()
            if done:
                break

        video.close()

    def viz(self, phy: Physics, is_planning=False, excluded_geom_names=[]):
        m = phy.m
        d = phy.d
        for geom_id in range(m.ngeom):
            d_geom = d.geom(geom_id)
            m_geom = m.geom(geom_id)
            geom_type = m_geom.type
            geom_bodyid = m_geom.bodyid
            d_body = d.body(geom_bodyid)

            if m_geom.name in excluded_geom_names:
                continue

            geom_xmat = d_geom.xmat.reshape(3, 3)
            geom_transform = np.eye(4)
            geom_transform[0:3, 0:3] = geom_xmat
            geom_transform[0:3, 3] = d_geom.xpos
            geom_xpos = d_geom.xpos

            body_xmat = d_body.xmat.reshape(3, 3)
            body_transform = np.eye(4)
            body_transform[0:3, 0:3] = body_xmat
            body_transform[0:3, 3] = d_body.xpos
            c = m_geom.rgba[:3]
            alpha = m_geom.rgba[3]
            if is_planning:
                alpha *= 0.3

            if geom_type == mjtGeom.mjGEOM_BOX:
                fulL_size = (2 * m_geom.size)
                box = Box(geom_xpos, *fulL_size, c=c, alpha=alpha)
                if geom_id not in self.actor_map:
                    self.plotter += box
                    self.actor_map[geom_id] = box
                box = self.actor_map[geom_id]
                box.apply_transform(geom_transform)
            elif geom_type == mjtGeom.mjGEOM_CYLINDER:
                cy = Cylinder(geom_xpos, m_geom.size[0], 2 * m_geom.size[1], c=c, alpha=alpha)
                if geom_id not in self.actor_map:
                    self.plotter += cy
                    self.actor_map[geom_id] = cy
                cy = self.actor_map[geom_id]
                cy.apply_transform(geom_transform)
            elif geom_type == mjtGeom.mjGEOM_SPHERE:
                sphere = Sphere(geom_xpos, m_geom.size[0], c=c, alpha=alpha)
                if geom_id not in self.actor_map:
                    self.plotter += sphere
                    self.actor_map[geom_id] = sphere
                sphere = self.actor_map[geom_id]
                sphere.apply_transform(geom_transform)
            elif geom_type == mjtGeom.mjGEOM_CAPSULE:
                start = geom_xpos - m_geom.size[1] * geom_xmat[:, 2]
                end = geom_xpos + m_geom.size[1] * geom_xmat[:, 2]
                if geom_id not in self.actor_map:
                    cy = Cylinder(geom_xpos, m_geom.size[0], 2 * m_geom.size[1], c=c, alpha=alpha)
                    s1 = Sphere(start, m_geom.size[0], c=c, alpha=alpha)
                    s2 = Sphere(end, m_geom.size[0], c=c, alpha=alpha)
                    self.plotter += cy
                    self.plotter += s1
                    self.plotter += s2
                    self.actor_map[geom_id] = [cy, s1, s2]
                cy, s1, s2 = self.actor_map[geom_id]
                cy.apply_transform(geom_transform)
                s1.pos(start)
                s2.pos(end)
            elif geom_type == mjtGeom.mjGEOM_MESH:
                mesh_name = mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m_geom.dataid)
                # skip the model prefix, e.g. val/my_mesh
                if '/' in mesh_name:
                    mesh_name = mesh_name.split("/")[1]
                mesh_name = Path(self.mj_xml_parser.get_mesh(mesh_name))
                if mesh_name not in self.mesh_cache:
                    self.load_and_cache_mesh(mesh_name)

                if geom_id not in self.actor_map:
                    mesh = self.mesh_cache[mesh_name].clone()
                    self.plotter += mesh
                    self.actor_map[geom_id] = mesh
                mesh = self.actor_map[geom_id]
                if is_planning:
                    mesh.alpha(0.5)
                mesh.apply_transform(body_transform)

    def get_object_actors(self, mjobject: MjObject):
        actors = []
        for geom_id in mjobject.geom_indices:
            actor = self.actor_map[geom_id]
            if isinstance(actor, list):
                actors.extend(actor)
            else:
                actors.append(actor)
        return actors

    def get_actor(self, phy, geom_name: str) -> Points:
        goem_id = phy.m.geom(geom_name).id
        return self.actor_map[goem_id]

    def num_frames_for_spin(self, seconds_per_spin, n_spins):
        return int(seconds_per_spin * self.fps * n_spins)

    def load_and_cache_mesh(self, mesh_name):
        extensions = [".obj", ".stl", ".glb"]
        for ext in extensions:
            mesh_file = mesh_name.stem + ext
            mesh_file = Path.cwd() / "models" / "meshes" / mesh_file
            if not mesh_file.exists():
                continue
            mesh = load(str(mesh_file))
            if mesh is None:
                continue
            self.mesh_cache[mesh_name] = mesh
            return
        raise RuntimeError(f"Mesh {mesh_name} not found")


def load_frame_from_npy(frame_idx, qpos_filename, scenario):
    outdir, phy, qpos, skeletons, trial_idx = load_from_npy(qpos_filename, scenario)
    # Load the given frame and render it with mjvedo
    set_phy_to_frame(phy, qpos, frame_idx)
    return outdir, phy, qpos, skeletons, trial_idx


def load_from_npy(qpos_filename, scenario):
    outdir = qpos_filename.parent
    qpos = np.load(qpos_filename)
    trial_idx = int(qpos_filename.stem.split('_')[-3])
    phy, sdf_path, skeletons = load_phy_and_skeletons(trial_idx, scenario)
    return outdir, phy, qpos, skeletons, trial_idx


def set_phy_to_frame(phy, qpos, frame_idx):
    phy.d.qpos = qpos[frame_idx]
    mujoco.mj_forward(phy.m, phy.d)
