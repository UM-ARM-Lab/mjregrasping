#!/usr/bin/env python3
import mujoco
import numpy as np

from arc_utilities.tf2wrapper import TF2Wrapper
from arm_rviz.rviz_animation_controller import RvizAnimationController
from mjregrasping.mujoco_visualizer import MujocoVisualizer
from mjregrasping.rollout import parallel_rollout
import argparse

import rospy


def main():
    np.set_printoptions(suppress=True, precision=5)

    parser = argparse.ArgumentParser()
    parser.add_argument("xml_path", type=str)
    parser.add_argument("--n-time", type=int, default=20)

    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)

    n_time = args.n_time
    n_samples = 100

    controls_samples = np.random.uniform(-0.4, 0.4, size=[n_samples, 1, model.nu])
    controls_samples = np.tile(controls_samples, [1, n_time, 1])
    # controls_samples = np.zeros([n_samples, n_time, model.nu])
    # controls_samples[:, :, 1] = 0.4

    states = parallel_rollout(model, data, controls_samples)

    rospy.init_node("viz_rollouts")

    tfw = TF2Wrapper()
    mjviz = MujocoVisualizer(tfw)

    trajs_viz = RvizAnimationController(n_time_steps=n_samples, ns='trajs')
    time_viz = RvizAnimationController(n_time_steps=n_time, ns='time')
    while not trajs_viz.done:
        traj_idx = trajs_viz.t()
        time_viz.reset()
        while not time_viz.done:
            t = time_viz.t()

            data.qpos[:] = states[traj_idx, t, :]
            mujoco.mj_forward(model, data)
            mjviz.viz(model, data)

            time_viz.step()
        trajs_viz.step()



if __name__ == "__main__":
    main()
