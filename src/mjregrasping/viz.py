import numpy as np
import pysdf_tools
import rerun as rr
from matplotlib import cm
from matplotlib.colors import to_rgba

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.params import Params
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rviz import MjRViz, plot_sphere_rviz, plot_lines_rviz, plot_ring_rviz, plot_arrows_rviz, \
    plot_points_rviz
from visualization_msgs.msg import MarkerArray


def make_viz(scenario):
    tfw = TF2Wrapper()
    mjviz = MjRViz(scenario.xml_path, tfw)
    p = Params()
    return Viz(rviz=mjviz, mjrr=MjReRun(scenario.xml_path), tfw=tfw, p=p)


class Viz:

    def __init__(self, rviz: MjRViz, mjrr: MjReRun, tfw: TF2Wrapper, p: Params):
        self.rviz = rviz
        self.mjrr = mjrr
        self.pubs = None
        self.tfw = tfw
        self.p = p

        self.markers_pub = rospy.Publisher("markers", MarkerArray, queue_size=10)

    def sphere(self, ns: str, position, radius, color, idx=0, frame_id='world'):
        if self.p.rviz:
            plot_sphere_rviz(pub=self.markers_pub, position=position, radius=radius, frame_id=frame_id, color=color,
                             label=f'{ns}', idx=idx)
        if self.p.rr:
            rr_color = to_rgba(color)
            rr.log_point(f'{ns}/{idx}', position, color=rr_color, radius=radius)

    def points(self, ns: str, positions, color, idx=0, radius=0.01, frame_id='world'):
        if self.p.rviz:
            plot_points_rviz(self.markers_pub, positions, idx=idx, label=f'{ns}', color=color, s=radius / 0.01,
                             frame_id=frame_id)
        if self.p.rr:
            rr_color = to_rgba(color)
            rr.log_points(f'{ns}/{idx}', positions, colors=rr_color, radii=radius)

    def lines(self, positions, ns: str, idx: int, scale: float, color, frame_id='world'):
        if self.p.rviz:
            plot_lines_rviz(
                pub=self.markers_pub,
                positions=positions,
                label=f'{ns}',
                idx=idx,
                color=color,
                frame_id=frame_id,
                scale=scale,
            )

        if self.p.rr:
            rr_color = to_rgba(color)
            rr.log_line_strip(
                entity_path=f'{ns}/{idx}',
                positions=positions,
                color=rr_color,
                stroke_width=scale,
            )

    def ring(self, ring_position, ring_z_axis, radius):
        if self.p.rviz:
            plot_ring_rviz(self.markers_pub, ring_position, ring_z_axis, radius)
        if self.p.rr:
            # TODO: also show in rerun
            pass

    def tf(self, translation, quat_xyzw, parent='world', child='gripper_point_goal'):
        if self.p.rviz:
            self.tfw.send_transform(translation, quat_xyzw, parent=parent, child=child)
        if self.p.rr:
            # TODO: also show in rerun
            pass

    def viz(self, phy: Physics, is_planning: bool = False):
        if self.p.rviz:
            if is_planning:
                if self.p.viz_planning:
                    self.rviz.viz(phy, is_planning, alpha=self.p.is_planning_alpha)
            else:
                self.rviz.viz(phy, is_planning, alpha=1.0)
        if self.p.rr:
            self.mjrr.viz(phy, is_planning)

    def sdf(self, sdf: pysdf_tools.SignedDistanceField, frame_id, idx):
        # NOTE: VERY SLOW!!! only use for debugging
        if self.p.rr:
            self.mjrr.sdf(sdf, frame_id, idx)
        if self.p.rviz:
            pass
        #     self.rviz.sdf(sdf, frame_id, idx)

    def arrow(self, ns, start, direction, color, idx=0, frame_id='world', s=1):
        if self.p.rviz:
            plot_arrows_rviz(self.markers_pub, [start], [direction], ns, idx, color, frame_id, s)
        if self.p.rr:
            rr_color = to_rgba(color)
            rr.log_arrow(f'{ns}/{idx}', start, direction, color=rr_color)
