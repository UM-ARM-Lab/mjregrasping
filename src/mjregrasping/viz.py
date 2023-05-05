import mujoco

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.mujoco_visualizer import MjRViz
from mjregrasping.params import Params
from mjregrasping.rerun_visualizer import MjReRun
from visualization_msgs.msg import MarkerArray


class Viz:

    def __init__(self, rviz: MjRViz, mjrr: MjReRun, tfw: TF2Wrapper, p: Params):
        self.rviz = rviz
        self.mjrr = mjrr
        self.pubs = None
        self.tfw = tfw
        self.p = p

        self.state = rospy.Publisher("state_viz", MarkerArray, queue_size=10)
        self.action = rospy.Publisher("action_viz", MarkerArray, queue_size=10)
        self.ee_path = rospy.Publisher("ee_path", MarkerArray, queue_size=10)
        self.goal = rospy.Publisher("goal_markers", MarkerArray, queue_size=10)

    def sphere(self, ns: str, position, radius, frame_id, color):
        pass

    def lines(self, positions, ns: str, idx: int, scale: float, color):
        pass

    def tf(self, translation, quat_xyzw, parent='world', child='gripper_point_goal'):
        pass

    def viz(self, m: mujoco.MjModel, d: mujoco.MjData):
        if self.p.rviz:
            self.rviz.viz(m, d)
        if self.p.rr:
            self.mjrr.viz(m, d)
