import numpy as np

from mjregrasping.grasp_conversions import grasp_locations_to_xpos


def get_rope_points(phy):
    rope_points = phy.d.xpos[phy.o.rope.body_indices]
    end_point = grasp_locations_to_xpos(phy, np.ones(1))[0]
    rope_points = np.concatenate([rope_points, [end_point]])
    return rope_points
