import logging
import multiprocessing
import time
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

import mujoco
import rerun as rr

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.movie import MjMovieMaker
from mjregrasping.params import Params
from mjregrasping.physics import Physics
from mjregrasping.regrasp_mpc import RegraspMPC
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rviz import MjRViz
from mjregrasping.viz import Viz

logger = logging.getLogger(f'rosout.{__name__}')


class Runner:

    def __init__(self, xml_path):
        rr.init('regrasp_mpc_runner')
        rr.connect()

        self.xml_path = xml_path
        self.skeletons = self.get_skeletons()
        self.tfw = TF2Wrapper()
        self.mjviz = MjRViz(xml_path, self.tfw)
        self.p = Params()
        self.viz = Viz(rviz=self.mjviz, mjrr=MjReRun(xml_path), tfw=self.tfw, p=self.p)

        self.root = Path("results") / self.__class__.__name__
        self.root.mkdir(exist_ok=True, parents=True)

    def run(self, seeds, obstacle_name):
        for seed in seeds:
            m = mujoco.MjModel.from_xml_path(self.xml_path)
            d = mujoco.MjData(m)
            phy = Physics(m, d, obstacle_name)

            self.setup_scene(phy, self.viz)

            mov = MjMovieMaker(m)
            now = int(time.time())
            mov_path = self.root / f'seed_{seed}_{now}.mp4'
            logger.info(f"Saving movie to {mov_path}")
            mov.start(mov_path, fps=12)

            goal = self.make_goal(phy)

            with ThreadPoolExecutor(multiprocessing.cpu_count() - 1) as pool:
                self.viz.p.w_goal = 1.0
                self.viz.p.w_regrasp_point = 0.0
                self.viz.p.update()

                mpc = RegraspMPC(pool, phy.m.nu, self.skeletons, goal, seed, self.viz, mov)
                mpc.run(phy)
                mpc.close()

    def make_goal(self):
        raise NotImplementedError()
        # goal = ObjectPointGoal(dfield=None,
        #                        viz=self.viz,
        #                        goal_point=goal_point,
        #                        body_idx=goal_body_idx,
        #                        goal_radius=0.05,
        #                        )

    def setup_scene(self, phy: Physics, viz: Viz):
        pass

    def get_skeletons(self):
        raise NotImplementedError()
