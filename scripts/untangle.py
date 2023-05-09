#!/usr/bin/env python3
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import rerun as rr

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.body_with_children import Objects
from mjregrasping.goals import ObjectPointGoal
from mjregrasping.grasping import activate_eq
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.movie import MjMovieMaker
from mjregrasping.params import Params
from mjregrasping.regrasp_mpc import RegraspMPC
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rollout import control_step, DEFAULT_SUB_TIME_S
from mjregrasping.rviz import MjRViz
from mjregrasping.viz import Viz

logger = logging.getLogger(f'rosout.{__name__}')


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rr.init('mjregrasping')
    rr.connect()
    rospy.init_node("untangle")
    xml_path = "models/untangle_scene.xml"
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    tfw = TF2Wrapper()
    mjviz = MjRViz(xml_path, tfw)
    mujoco.mj_forward(m, d)
    mjviz.viz(m, d)
    p = Params()
    viz = Viz(rviz=mjviz, mjrr=MjReRun(), tfw=tfw, p=p)

    goal_rng = np.random.RandomState(0)

    root = Path("results")
    root.mkdir(exist_ok=True)

    for seed in range(p.n_seeds):
        mov = MjMovieMaker(m, "rack1")
        mov_path = root / f'untangle_{seed}.mp4'
        mov.start(mov_path, fps=12)

        m = mujoco.MjModel.from_xml_path("models/untangle_scene.xml")
        objects = Objects(m)
        d = mujoco.MjData(m)

        # setup_untangled_scene(m, d, mjviz)
        setup_tangled_scene(m, d, viz)

        goal = ObjectPointGoal(model=m,
                               viz=viz,
                               goal_point=np.array([0.78, 0.04, 1.27]) + goal_rng.uniform(-0.05, 0.05, size=3),
                               body_idx=-1,
                               goal_radius=0.05,
                               objects=objects)

        with ThreadPoolExecutor(multiprocessing.cpu_count() - 1) as pool:
            mpc = RegraspMPC(m, pool, viz, goal, objects=objects, seed=seed, mov=mov)
            result = mpc.run(d)
            logger.info(f"{seed=} {result=}")


def setup_tangled_scene(m, d, viz):
    robot_q1 = np.array([
        -0.7, 0.1,  # torso
        -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
        0, 0,  # left gripper
        0.0, -0.2, 0, -0.30, 0, -0.2, 0,  # right arm
        0, 0,  # right gripper
    ])
    pid_to_joint_config(viz, m, d, robot_q1, sub_time_s=DEFAULT_SUB_TIME_S)
    robot_q2 = np.array([
        -0.5, 0.4,  # torso
        -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
        0, 0,  # left gripper
        1.2, -0.2, 0, -0.90, 0, -0.2, 0,  # right arm
        0, 0,  # right gripper
    ])
    pid_to_joint_config(viz, m, d, robot_q2, sub_time_s=DEFAULT_SUB_TIME_S)
    activate_and_settle(m, d, viz, sub_time_s=DEFAULT_SUB_TIME_S)


def setup_untangled_scene(m, d, viz):
    activate_and_settle(m, d, viz, sub_time_s=DEFAULT_SUB_TIME_S)


def activate_and_settle(m, d, viz, sub_time_s):
    # Activate the connect constraint between the rope and the gripper to
    activate_eq(m, 'right')
    # settle
    for _ in range(25):
        viz.viz(m, d)
        control_step(m, d, np.zeros(m.nu), sub_time_s=sub_time_s)
        rospy.sleep(0.01)


if __name__ == "__main__":
    main()
