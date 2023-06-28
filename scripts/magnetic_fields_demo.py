import numpy as np
import rerun as rr
from numpy.linalg import norm

from mjregrasping.magnetic_fields import skeleton_field_dir


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    np.seterr(all='raise')
    rr.init("threading_cost_demo")
    rr.connect()
    rr.log_view_coordinates("world", up="+Z", timeless=True)
    rr.log_arrow('world_x', [0, 0, 0], [1, 0, 0], color=(255, 0, 0), width_scale=0.02, timeless=True)
    rr.log_arrow('world_y', [0, 0, 0], [0, 1, 0], color=(0, 255, 0), width_scale=0.02, timeless=True)
    rr.log_arrow('world_z', [0, 0, 0], [0, 0, 1], color=(0, 0, 255), width_scale=0.02, timeless=True)

    skeleton = np.array([
        [0.5, -0.2, 0],
        [0.5, 0.2, 0],
        [0.5, 0.2, 1],
        [0.0, 0.2, 1],
        [0.0, 0.2, 0.5],
        [0.0, -0.2, 0.5],
        [0.0, -0.2, 1],
        [0.5, -0.2, 1],
        [0.5, -0.2, 0],
    ])

    ring_position = np.array([0, 1, 1])
    ring_z_axis = np.array([0, 1, 0])
    R = 0.5  # radius
    max_norm = 0.2

    n1 = 50
    n2 = 25
    ys = np.linspace(0.35, 0.45, n1)
    zs = np.linspace(0.75, 1.25, n2)
    rr.log_line_strip('skeleton', skeleton, color=(0, 255, 0, 255), timeless=True)
    Ys, Zs = np.meshgrid(ys, zs, indexing='xy')
    xs = np.ones(n1 * n2)
    p = np.stack([xs, Ys.flatten(), Zs.flatten()], axis=1)

    for t in range(50000):
        # b = compute_threading_dir(ring_position, ring_z_axis, R, p)
        b = skeleton_field_dir(skeleton, p)
        p = p + b * 0.05

        if t % 5 == 0:
            rr.log_points('field', p, colors=(0, 0, 255, 200), radii=0.01)
            # for i, (p_i, b_i) in enumerate(zip(p, b)):
            #     if norm(b_i) > max_norm:
            #         b_i = b_i / norm(b_i) * max_norm
            #     rr.log_arrow(f'field/{i}', p_i, b_i, color=(255, 0, 0, 255))


if __name__ == '__main__':
    main()
