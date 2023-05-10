import imageio
import mujoco
import numpy as np


class MjMovieMaker:
    def __init__(self, m: mujoco.MjModel, trackbody_name: str, w=1280, h=720):
        """ Initialize a movie maker for a mujoco model """
        self.m = m
        self.gl_ctx = mujoco.GLContext(w, h)
        self.gl_ctx.make_current()
        self.con = mujoco.MjrContext(m, 1)
        # TODO: disable shadows
        self.scene = mujoco.MjvScene(m, maxgeom=500, shadow=0)
        self.viewport = mujoco.MjrRect(0, 0, w, h)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = self.m.body(trackbody_name).id
        self.cam.distance = 3.0
        self.cam.azimuth = 35
        self.img = np.zeros([h, w, 3], dtype=np.uint8)
        self.writer = None

    def render(self, d: mujoco.MjData):
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
        mujoco.mjr_readPixels(rgb=self.img, depth=None, viewport=self.viewport, con=self.con)
        self.writer.append_data(np.flipud(self.img))

    def start(self, filename, fps):
        """ Reset the movie maker """
        self.writer = imageio.get_writer(filename, fps=fps)

    def close(self):
        """ Save the movie as a .mp4 file """
        self.writer.close()
        # reset for next time
