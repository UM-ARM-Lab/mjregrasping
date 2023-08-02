from copy import deepcopy

import imageio
import mujoco
import numpy as np
from mujoco._structs import _MjModelCameraViews


class MjMovieMaker:
    def __init__(self, m: mujoco.MjModel, w=1280, h=720):
        """ Initialize a movie maker for a mujoco model """
        self.m = m
        self.r = MjRenderer(m, w, h)
        self.writer = None

    def render(self, d: mujoco.MjData):
        """ Render the current mujoco scene and store the resulting image """
        self.writer.append_data(self.r.render(d))

    def start(self, filename, fps):
        """ Reset the movie maker """
        self.writer = imageio.get_writer(filename, fps=fps)

    def close(self):
        """ Save the movie as a .mp4 file """
        self.writer.close()


class MjRenderer:
    def __init__(self, m: mujoco.MjModel, w=1280, h=720, cam: mujoco.MjvCamera = None):
        """ Initialize a movie maker for a mujoco model """
        self.m = m
        self.w = w
        self.h = h
        self.gl_ctx = mujoco.GLContext(w, h)
        self.gl_ctx.make_current()
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

    def render(self, d: mujoco.MjData, depth=False):
        """ Render the current mujoco scene and store the resulting image """
        catmask = mujoco.mjtCatBit.mjCAT_ALL
        mujoco.mjv_updateScene(self.m,
                               d,
                               opt=mujoco.MjvOption(),
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
