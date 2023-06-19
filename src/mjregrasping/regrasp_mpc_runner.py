import logging
import multiprocessing
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

import mujoco
import rerun as rr

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.body_with_children import Objects
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
        rospy.init_node("regrasp_mpc_runner")

        self.xml_path = xml_path
        self.tfw = TF2Wrapper()
        self.mjviz = MjRViz(xml_path, self.tfw)
        self.p = Params()
        self.viz = Viz(rviz=self.mjviz, mjrr=MjReRun(xml_path), tfw=self.tfw, p=self.p)

        self.root = Path("results") / self.__class__.__name__
        self.root.mkdir(exist_ok=True, parents=True)

    def run(self, seeds, obstacle_name):
        for seed in seeds:
            m = mujoco.MjModel.from_xml_path(self.xml_path)
            objects = Objects(m, obstacle_name)
            d = mujoco.MjData(m)
            phy = Physics(m, d)

            self.setup_scene(phy, self.viz)

            mov = MjMovieMaker(m)
            mov_path = self.root / f'seed_{seed}.mp4'
            logger.info(f"Saving movie to {mov_path}")
            mov.start(mov_path, fps=12)

            goal = self.make_goal(phy, objects)

            with ThreadPoolExecutor(multiprocessing.cpu_count() - 1) as pool:
                from time import perf_counter
                t0 = perf_counter()
                mpc = RegraspMPC(pool=pool, mppi_nu=phy.m.nu, viz=self.viz, goal=goal, objects=objects, seed=seed,
                                 mov=mov)
                result = mpc.run(phy)
                logger.info(f'dt: {perf_counter() - t0:.4f}')
                logger.info(f"{seed=} {result=}")

    def make_goal(self, objects):
        raise NotImplementedError()
        # goal = ObjectPointGoal(dfield=None,
        #                        viz=self.viz,
        #                        goal_point=goal_point,
        #                        body_idx=goal_body_idx,
        #                        goal_radius=0.05,
        #                        objects=objects)

    def setup_scene(self, phy: Physics, viz: Viz):
        pass
