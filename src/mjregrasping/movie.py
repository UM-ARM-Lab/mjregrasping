from copy import deepcopy

import imageio
import mujoco
import numpy as np


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
    def __init__(self, m: mujoco.MjModel, w=1280, h=720):
        """ Initialize a movie maker for a mujoco model """
        self.m = m
        self.gl_ctx = mujoco.GLContext(w, h)
        self.gl_ctx.make_current()
        self.con = mujoco.MjrContext(m, 1)
        self.scene = mujoco.MjvScene(m, maxgeom=500)
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.viewport = mujoco.MjrRect(0, 0, w, h)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = 0
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
            img = self.depth
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
