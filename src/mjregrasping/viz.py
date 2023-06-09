import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.params import Params
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rviz import MjRViz, plot_sphere_rviz, plot_lines_rviz, plot_ring_rviz
from visualization_msgs.msg import MarkerArray


class Viz:

    def __init__(self, rviz: MjRViz, mjrr: MjReRun, tfw: TF2Wrapper, p: Params):
        self.rviz = rviz
        self.mjrr = mjrr
        self.pubs = None
        self.tfw = tfw
        self.p = p

        self.markers_pub = rospy.Publisher("markers", MarkerArray, queue_size=10)

    def sphere(self, ns: str, position, radius, frame_id, color, idx):
        plot_sphere_rviz(pub=self.markers_pub, position=position, radius=radius, frame_id=frame_id, color=color,
                         label=f'{ns}', idx=idx)
        # TODO: also show in rerun

    def lines(self, positions, ns: str, idx: int, scale: float, color):
        plot_lines_rviz(
            pub=self.markers_pub,
            positions=positions,
            label=f'{ns}',
            idx=idx,
            color=color,
            frame_id='world',
            scale=scale,
        )
        # TODO: also show in rerun

    def ring(self, ring_position, ring_z_axis, radius):
        plot_ring_rviz(self.markers_pub, ring_position, ring_z_axis, radius)

    def tf(self, translation, quat_xyzw, parent='world', child='gripper_point_goal'):
        self.tfw.send_transform(translation, quat_xyzw, parent=parent, child=child)
        # TODO: also show in rerun

    def viz(self, phy: Physics, is_planning: bool = False):
        if self.p.rviz:
            if is_planning:
                if self.p.viz_planning:
                    self.rviz.viz(phy, is_planning, alpha=self.p.is_planning_alpha)
            else:
                self.rviz.viz(phy, is_planning, alpha=1.0)
        if self.p.rr:
            self.mjrr.viz(phy)
