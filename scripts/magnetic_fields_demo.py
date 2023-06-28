import numpy as np
import rerun as rr

from mjregrasping.magnetic_fields import get_h_signature, discretize_path, make_ring_skeleton
from mjregrasping.viz_magnetic_fields import animate_field


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    np.seterr(all='raise')
    rr.init("threading_cost_demo")
    rr.connect()
    rr.log_view_coordinates("world", up="+Z", timeless=True)
    rr.log_arrow('world_x', [0, 0, 0], [1, 0, 0], color=(255, 0, 0), width_scale=0.02, timeless=True)
    rr.log_arrow('world_y', [0, 0, 0], [0, 1, 0], color=(0, 255, 0), width_scale=0.02, timeless=True)
    rr.log_arrow('world_z', [0, 0, 0], [0, 0, 1], color=(0, 0, 255), width_scale=0.02, timeless=True)

    h_signature_demo()


def h_signature_demo():
    skeleton = np.array([
        [0.5, -0.2, 0],
        [0.5, 0.2, 0],
        [0.5, 0.2, 1],
        [0.1, 0.2, 1],
        [0.1, 0.2, 0.5],
        [0.1, -0.2, 0.5],
        [0.1, -0.2, 1],
        [0.5, -0.2, 1],
        [0.5, -0.2, 0],
    ])

    # Computing H-signature of paths
    rr.log_line_strip('skeleton', skeleton, color=(0, 255, 0, 255), timeless=True)

    z = 0.23
    y = 0
    path1 = discretize_path(np.array([[0, y, z], [1., y, z], [1., y, z + 0.2], [0, y, z + 0.2], [0, y, z]]))

    # integrate the magnetic field along the trajectory
    from time import perf_counter
    t0 = perf_counter()
    h = get_h_signature(path1, skeleton)
    print(f'dt: {perf_counter() - t0:.4f}')
    rr.log_line_strip('path1', path1, ext={'h': h})


def anim_ring_demo():
    skeleton = make_ring_skeleton(position=np.array([0.8, 1, 1]), z_axis=np.array([1, 0, 0]), radius=0.5)
    animate_field(skeleton)


if __name__ == '__main__':
    main()
