import time

import hjson
import numpy as np
import rerun as rr

from mjregrasping.magnetic_fields import get_h_signature, discretize_path, make_ring_skeleton, load_skeletons
from mjregrasping.rerun_visualizer import log_skeletons
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
    # skeletons = get_example_skeletons()
    skeletons = load_skeletons('models/computer_rack_skeleton.hjson')

    # Computing H-signature of paths
    log_skeletons(skeletons, color=(0, 255, 0, 255), timeless=True)

    z = 0.43
    all_hs = []
    for y in np.linspace(0.01, 1, 10):
        # integrate the magnetic field along the trajectory
        from time import perf_counter
        t0 = perf_counter()
        path = np.array([[0, y, z], [1., y, z], [1., y, z + 0.2], [0, y, z + 0.2], [0, y, z]])
        h = get_h_signature(path, skeletons)
        all_hs.append(h)
        print(f'{y=:.2f} {h=} computing H-signature: {perf_counter() - t0:.4f}s')
        rr.log_line_strip('path', path, ext={'hs': str(h)})

        time.sleep(0.1)

    print(np.unique(all_hs, axis=0))


def get_example_skeletons():
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
    return {
        'example': skeleton,
    }


def anim_ring_demo():
    skeleton = make_ring_skeleton(position=np.array([0.8, 1, 1]), z_axis=np.array([1, 0, 0]), radius=0.5)
    animate_field(skeleton)


if __name__ == '__main__':
    main()
