import json
import logging
from copy import deepcopy
from pathlib import Path

import glfw
import imageio
import mujoco
import numpy as np
from mujoco import GLContext
from mujoco._structs import _MjModelCameraViews


class MjMovieMaker:
    def __init__(self, gl_ctx: GLContext, m: mujoco.MjModel):
        """ Initialize a movie maker for a mujoco model """
        self.m = m
        self.r = MjRenderer(gl_ctx, m)
        self.writer = None
        self.filename = None
        self.qpos_list = []

    def render(self, d: mujoco.MjData):
        """ Render the current mujoco scene and store the resulting image and qpos """
        self.qpos_list.append(d.qpos.copy())
        self.writer.append_data(self.r.render(d))

    def start(self, filename: Path):
        """ Set up the writer and filenames  """
        fps = int(1 / self.m.opt.timestep)
        self.filename = filename
        self.writer = imageio.get_writer(filename, fps=fps)

    def close(self, metrics):
        """ Finish the movie file and save the qpos data as .npy """
        self.writer.close()
        method = metrics.get('method', 'method_missing')
        success = metrics.get('success', 'success_missing')
        d = self.filename.parent
        stem = self.filename.stem
        qpos_path = d / f"{stem}_{method}_qpos.npy"
        metrics_path = d / f"{stem}_{method}_{success}.json"
        with qpos_path.open('wb') as f:
            np.save(f, np.array(self.qpos_list))
        with metrics_path.open('w') as f:
            json.dump(metrics, f, indent=2)


class MjRenderer:
    def __init__(self, gl_ctx: GLContext, m: mujoco.MjModel, cam: mujoco.MjvCamera = None):
        """ Initialize a movie maker for a mujoco model """
        w, h = glfw.get_window_size(gl_ctx._context)
        self.w = w
        self.h = h
        self.m = m
        self.con = mujoco.MjrContext(m, 1)
        self.scene = mujoco.MjvScene(m, maxgeom=500)
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.viewport = mujoco.MjrRect(0, 0, w, h)
        if cam is None:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = m.cam("mycamera").id
        self.cam = cam
        self.rgb = np.zeros([h, w, 3], dtype=np.uint8)
        self.depth = np.zeros([h, w], dtype=np.float32)
        self.opt = mujoco.MjvOption()

    def render(self, d: mujoco.MjData, depth=False):
        """ Render the current mujoco scene and store the resulting image """
        catmask = mujoco.mjtCatBit.mjCAT_ALL
        mujoco.mjv_updateScene(self.m,
                               d,
                               opt=self.opt,
                               pert=None,
                               cam=self.cam,
                               catmask=catmask,
                               scn=self.scene)

        # render the scene from a mujoco camera and store the result in img
        mujoco.mjr_render(self.viewport, self.scene, self.con)
        mujoco.mjr_readPixels(rgb=self.rgb, depth=self.depth, viewport=self.viewport, con=self.con)

        if depth:
            extent = self.m.stat.extent
            near = self.m.vis.map.znear * extent
            far = self.m.vis.map.zfar * extent
            # Convert from [0 1] to depth in meters, see links below:
            # http://stackoverflow.com/a/6657284/1461210
            # https://www.khronos.org/opengl/wiki/Depth_Buffer_Precision
            img = near / (1 - self.depth * (1 - near / far))
        else:
            img = self.rgb

        return np.flipud(img)

    def render_with_flags(self, d: mujoco.MjData, flags):
        before_flags = deepcopy(self.scene.flags)
        for flag, value in flags.items():
            self.scene.flags[flag] = value
        img = self.render(d)
        self.scene.flags = before_flags
        return img


class MjRGBD:

    def __init__(self, mcam: _MjModelCameraViews, renderer: MjRenderer):
        self.mcam = mcam
        self.r = renderer
        self.w = renderer.w
        self.h = renderer.h
        self.fpx = 0.5 * self.h / np.tan(mcam.fovy[0] * np.pi / 360)
        self.cx = self.w / 2
        self.cy = self.h / 2
