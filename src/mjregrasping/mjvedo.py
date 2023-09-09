from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
from mujoco import mjtGeom, mj_id2name
from vedo import Plotter, Video, ProgressBarWidget, Box, Cylinder, Sphere, load, Points, Light

from mjregrasping.physics import Physics
from mjregrasping.rviz import MujocoXmlExpander

COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']


class MjVedo:

    def __init__(self, xml_path, fps=60):
        self.mj_xml_parser = MujocoXmlExpander(xml_path)
        self.fps = fps

        self.plotter = Plotter(title="drone_example", axes=1)

        self.mesh_cache = {}
        # maps from mujoco geom IDs to Vedo actors
        self.actor_map = {}

    def record(self, filename, anim_func: Callable, num_frames: int):
        pb = ProgressBarWidget(num_frames)
        self.plotter += pb

        video = Video(filename, fps=self.fps, backend="ffmpeg")

        for t in range(num_frames):
            anim_func(t, self.plotter)
            # self.plotter.render()
            self.plotter.show(interactive=False)
            video.add_frame()
            pb.update()

        video.close()

    def viz(self, phy: Physics, is_planning=False):
        m = phy.m
        d = phy.d
        for geom_id in range(m.ngeom):
            d_geom = d.geom(geom_id)
            m_geom = m.geom(geom_id)
            geom_type = m_geom.type
            geom_bodyid = m_geom.bodyid
            d_body = d.body(geom_bodyid)

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
                box.apply_transform(geom_transform)
                self.plotter += box
                self.actor_map[geom_id] = box
            elif geom_type == mjtGeom.mjGEOM_CYLINDER:
                cy = Cylinder(geom_xpos, m_geom.size[0], 2 * m_geom.size[1], c=c, alpha=alpha)
                cy.apply_transform(geom_transform)
                self.plotter += cy
                self.actor_map[geom_id] = cy
            elif geom_type == mjtGeom.mjGEOM_SPHERE:
                sphere = Sphere(geom_xpos, m_geom.size[0], c=c, alpha=alpha)
                sphere.apply_transform(geom_transform)
                self.plotter += sphere
                self.actor_map[geom_id] = sphere
            elif geom_type == mjtGeom.mjGEOM_CAPSULE:
                cy = Cylinder(geom_xpos, m_geom.size[0], 2 * m_geom.size[1], c=c, alpha=alpha)
                cy.apply_transform(geom_transform)
                self.plotter += cy
                start = geom_xpos - m_geom.size[1] * geom_xmat[:, 2]
                end = geom_xpos + m_geom.size[1] * geom_xmat[:, 2]
                s1 = Sphere(start, m_geom.size[0], c=c, alpha=alpha)
                s1.apply_transform(geom_transform)
                self.plotter += s1
                s2 = Sphere(end, m_geom.size[0], c=c, alpha=alpha)
                s2.apply_transform(geom_transform)
                self.plotter += s2
                self.actor_map[geom_id] = [cy, s1, s2]
            elif geom_type == mjtGeom.mjGEOM_MESH:
                mesh_name = mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m_geom.dataid)
                # skip the model prefix, e.g. val/my_mesh
                if '/' in mesh_name:
                    mesh_name = mesh_name.split("/")[1]
                mesh_name, mesh_scale = self.mj_xml_parser.get_mesh(mesh_name)
                mesh_name = Path(mesh_name)
                if mesh_name not in self.mesh_cache:
                    self.load_and_cache_mesh(mesh_name)

                mesh = self.mesh_cache[mesh_name].clone()
                mesh.apply_transform(body_transform)
                self.plotter += mesh
                self.actor_map[geom_id] = mesh
        for light_id in range(m.nlight):
            light = m.light(light_id)
            if light.mode[0] == 2:
                vlight = Light(light.pos, intensity=0.4)
                # self.plotter += vlight

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
