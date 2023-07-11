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
from mjregrasping.rviz import MjRViz, plot_sphere_rviz, plot_lines_rviz, plot_ring_rviz
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

    def sphere(self, ns: str, position, radius, frame_id, color, idx):
        if self.p.rviz:
            plot_sphere_rviz(pub=self.markers_pub, position=position, radius=radius, frame_id=frame_id, color=color,
                             label=f'{ns}', idx=idx)
        if self.p.rr:
            rr_color = to_rgba(color)
            rr.log_point(f'{ns}/{idx}', position, color=rr_color, radius=radius)

    def lines(self, positions, ns: str, idx: int, scale: float, color):
        if self.p.rviz:
            plot_lines_rviz(
                pub=self.markers_pub,
                positions=positions,
                label=f'{ns}',
                idx=idx,
                color=color,
                frame_id='world',
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
            origin = sdf.GetOriginTransform().translation()
            points = []
            colors = []
            for x_i in range(0, sdf.GetNumXCells(), 1):
                for y_i in range(0, sdf.GetNumYCells(), 1):
                    for z_i in [54]:  # range(0, sdf.GetNumZCells(), 10):
                        p = origin + np.array([x_i, y_i, z_i]) * sdf.GetResolution()
                        points.append(p)
                        sdf_value = sdf.GetValueByIndex(x_i, y_i, z_i)[0]
                        colors.append([1, 0, 0, 1.0] if sdf_value < 0 else [0, 1, 0, 1.0])
            rr.log_points(f'sdf/{idx}', positions=points, colors=colors, radii=sdf.GetResolution() / 2)
        if self.p.rviz:
            pass
        #     self.rviz.sdf(sdf, frame_id, idx)
