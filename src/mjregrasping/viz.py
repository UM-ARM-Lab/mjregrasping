from typing import Dict

import numpy as np
import pysdf_tools
import rerun as rr
from matplotlib.colors import to_rgba

import ros_numpy
import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun, log_skeletons
from mjregrasping.rviz import MjRViz, plot_sphere_rviz, plot_lines_rviz, plot_ring_rviz, plot_arrows_rviz, \
    plot_points_rviz
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray


def make_viz(scenario):
    tfw = TF2Wrapper()
    mjviz = MjRViz(scenario.xml_path, tfw)
    return Viz(rviz=mjviz, mjrr=MjReRun(scenario.xml_path))


class Viz:

    def __init__(self, rviz: MjRViz, mjrr: MjReRun):
        self.rviz = rviz
        self.mjrr = mjrr
        self.pubs = None

        self.markers_pub = rospy.Publisher("markers", MarkerArray, queue_size=10)
        self.fig_pub = rospy.Publisher("fig", Image, queue_size=10)

    def fig(self, fig):
        image = plt_fig_to_img_np(fig)

        msg = ros_numpy.msgify(Image, image, encoding='rgb8')
        msg.header.stamp = rospy.Time.now()

        self.fig_pub.publish(msg)

    def sphere(self, ns: str, position, radius, color, idx=0, frame_id='world'):
        if self.rviz:
            plot_sphere_rviz(pub=self.markers_pub, position=position, radius=radius, frame_id=frame_id, color=color,
                             label=f'{ns}', idx=idx)
        if self.mjrr:
            rr_color = to_rgba(color)
            rr.log_point(f'{ns}/{idx}', position, color=rr_color, radius=radius)

    def points(self, ns: str, positions, color, idx=0, radius=0.01, frame_id='world'):
        if self.rviz:
            plot_points_rviz(self.markers_pub, positions, idx=idx, label=f'{ns}', color=color, s=radius / 0.01,
                             frame_id=frame_id)
        if self.mjrr:
            rr_color = to_rgba(color)
            rr.log_points(f'{ns}/{idx}', positions, colors=rr_color, radii=radius)

    def lines(self, positions, ns: str, idx: int, scale: float, color, frame_id='world'):
        if self.rviz:
            plot_lines_rviz(
                pub=self.markers_pub,
                positions=positions,
                label=f'{ns}',
                idx=idx,
                color=color,
                frame_id=frame_id,
                scale=scale,
            )

        if self.mjrr:
            rr_color = to_rgba(color)
            rr.log_line_strip(
                entity_path=f'{ns}/{idx}',
                positions=positions,
                color=rr_color,
                stroke_width=scale,
            )

    def ring(self, ring_position, ring_z_axis, radius):
        if self.rviz:
            plot_ring_rviz(self.markers_pub, ring_position, ring_z_axis, radius)
        if self.mjrr:
            # TODO: also show in rerun
            pass

    def skeletons(self, skeletons: Dict):
        if self.rviz:
            self.rviz.skeletons(skeletons)
        if self.mjrr:
            log_skeletons(skeletons)

    def viz(self, phy: Physics, is_planning: bool = False, detailed: bool = False):
        if self.rviz:
            if is_planning:
                self.rviz.viz(phy, is_planning, alpha=0.5)
            else:
                self.rviz.viz(phy, is_planning, alpha=1.0)
        if self.mjrr:
            self.mjrr.viz(phy, is_planning, detailed)

    def sdf(self, sdf: pysdf_tools.SignedDistanceField, frame_id='world', idx=0):
        # NOTE: VERY SLOW!!! only use for debugging
        if self.mjrr:
            self.mjrr.sdf(sdf)
        if self.rviz:
            pass

    def arrow(self, ns, start, direction, color, idx=0, frame_id='world', s=1):
        if self.rviz:
            plot_arrows_rviz(self.markers_pub, [start], [direction], ns, idx, color, frame_id, s)
        if self.mjrr:
            rr_color = to_rgba(color)
            rr.log_arrow(f'{ns}/{idx}', start, direction, color=rr_color)


def plt_fig_to_img_np(fig):
    canvas = fig.canvas
    canvas.draw()  # Draw the canvas, cache the renderer
    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)  # (H, W, 3)
    return image
