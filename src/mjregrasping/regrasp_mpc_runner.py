import logging
import multiprocessing
import pickle
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

import mujoco
import rerun as rr

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.body_with_children import Objects
from mjregrasping.dijsktra_field import make_dfield
from mjregrasping.goals import ObjectPointGoal, CombinedGoal
from mjregrasping.movie import MjMovieMaker
from mjregrasping.params import Params
from mjregrasping.physics import Physics
from mjregrasping.regrasp_mpc import RegraspMPC
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rviz import MjRViz
from mjregrasping.viz import Viz

logger = logging.getLogger(f'rosout.{__name__}')


class Runner:
    # Override these
    goal_point = None
    goal_body_idx = None
    obstacle_name = None
    dfield_path = None
    dfield_extents = None

    def __init__(self, xml_path):
        rr.init('regrasp_mpc_runner')
        rr.connect()
        rospy.init_node("regrasp_mpc_runner")

        self.xml_path = xml_path
        self.tfw = TF2Wrapper()
        self.mjviz = MjRViz(xml_path, self.tfw)
        self.p = Params()
        self.viz = Viz(rviz=self.mjviz, mjrr=MjReRun(xml_path), tfw=self.tfw, p=self.p)

        self.root = Path("results")
        self.root.mkdir(exist_ok=True)

    def run(self, seeds):
        for seed in seeds:
            m = mujoco.MjModel.from_xml_path(self.xml_path)
            objects = Objects(m, obstacle_name=self.obstacle_name)
            d = mujoco.MjData(m)
            phy = Physics(m, d)

            self.setup_scene(phy, self.viz)

            mov = MjMovieMaker(m)
            mov_path = self.root / f'untangle_{seed}.mp4'
            logger.info(f"Saving movie to {mov_path}")
            mov.start(mov_path, fps=12)

            # store and load from disk to save time?
            if self.dfield_path.exists():
                with self.dfield_path.open('rb') as f:
                    dfield = pickle.load(f)
            else:
                res = 0.02
                dfield = make_dfield(phy, self.dfield_extents, res, self.goal_point, objects)
                with self.dfield_path.open('wb') as f:
                    pickle.dump(dfield, f)

            goal = ObjectPointGoal(dfield=dfield,
                                   viz=self.viz,
                                   goal_point=self.goal_point,
                                   body_idx=self.goal_body_idx,
                                   goal_radius=0.05,
                                   objects=objects)
            goal = CombinedGoal(dfield, self.goal_point, 0.05, self.goal_body_idx, objects, self.viz)

            with ThreadPoolExecutor(multiprocessing.cpu_count() - 1) as pool:
                from time import perf_counter
                t0 = perf_counter()
                mpc = RegraspMPC(pool=pool, mppi_nu=phy.m.nu, viz=self.viz, goal=goal, objects=objects, seed=seed,
                                 mov=mov)
                result = mpc.run(phy)
                logger.info(f'dt: {perf_counter() - t0:.4f}')
                logger.info(f"{seed=} {result=}")

    def setup_scene(self, phy: Physics, viz: Viz):
        pass
