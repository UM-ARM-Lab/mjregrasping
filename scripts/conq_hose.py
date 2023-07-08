#!/usr/bin/env python3
from transformations import quaternion_from_euler
import mujoco
import numpy as np

from arc_utilities import ros_init
from mjregrasping.goals import ObjectPointGoal
from mjregrasping.grasping import activate_grasp
from mjregrasping.magnetic_fields import load_skeletons
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.mujoco_objects import Objects, Object
from mjregrasping.regrasp_mpc_runner import Runner
from mjregrasping.robot_data import conq
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.settle import settle


class ConqHose(Runner):

    def __init__(self):
        super().__init__("models/conq_hose_scene.xml")

    def setup_scene(self, phy, viz):
        # TODO: set up the scene so that the rope (hose) is at the attach point on the vacuum,
        #  put the hand near the end of the rope, activate the grasp constraint
        #  maybe set up the joint angles for the rope so that it is in a reasonable position.
        rope_xyz_q_indices = phy.o.rope.qpos_indices[:3]
        rope_quat_q_indices = phy.o.rope.qpos_indices[3:7]
        rope_ball_q_indices = phy.o.rope.qpos_indices[7:].reshape((-1, 4))
        phy.d.qpos[rope_xyz_q_indices] = np.array([2.0, 1.33, 0.25])
        phy.d.qpos[rope_quat_q_indices] = quaternion_from_euler(0, 0, np.pi)
        for j in range(rope_ball_q_indices.shape[0]):
            if 2 < j < 8:
                phy.d.qpos[rope_ball_q_indices[j]] = quaternion_from_euler(0, 0, np.deg2rad(18))
            else:
                phy.d.qpos[rope_ball_q_indices[j]] = quaternion_from_euler(0, 0, 0)
        mujoco.mj_forward(phy.m, phy.d)
        viz.viz(phy)
        attach_eq = phy.m.eq('attach')
        attach_eq.data[3:6] = 0
        attach_eq.active = True
        settle(phy, DEFAULT_SUB_TIME_S, viz, is_planning=False)

        # robot_q1 = np.array([
        #     0.0, 0,  # body x, y
        #     0.0,  # hand z
        #     0.6,  # gripper
        # ])
        # pid_to_joint_config(phy, viz, robot_q1, sub_time_s=DEFAULT_SUB_TIME_S)

    def make_goal(self, phy):
        goal_point = np.array([1.5, -1.5, 0.05])
        loc = 1
        goal = ObjectPointGoal(goal_point, 0.05, loc, self.viz)
        return goal

    def get_skeletons(self):
        return load_skeletons("models/hose_obstacles_skeleton.hjson")

    def get_objects(self, m):
        return Objects(m, "hose_obstacles", conq, "rope")


@ros_init.with_ros("conq_hose")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    runner = ConqHose()
    runner.run([1])


if __name__ == "__main__":
    main()
