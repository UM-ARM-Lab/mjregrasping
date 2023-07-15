import numpy as np
import rerun as rr

from mjregrasping.homotopy_utils import skeleton_field_dir


def animate_field(skeleton):
    n1 = 50
    n2 = 25
    ys = np.linspace(0.35, 0.45, n1)
    zs = np.linspace(0.75, 1.25, n2)
    rr.log_line_strip('skeleton', skeleton, color=(0, 255, 0, 255), timeless=True)
    Ys, Zs = np.meshgrid(ys, zs, indexing='xy')
    xs = np.ones(n1 * n2)
    p = np.stack([xs, Ys.flatten(), Zs.flatten()], axis=1)
    for t in range(50000):
        b = skeleton_field_dir(skeleton, p)
        p = p + b * 0.05

        if t % 25 == 0:
            rr.log_points('field', p, colors=(0, 0, 255, 200), radii=0.01)
            # for i, (p_i, b_i) in enumerate(zip(p, b)):
            #     rr.log_arrow(f'field/{i}', p_i, b_i, color=(255, 0, 0, 255))
