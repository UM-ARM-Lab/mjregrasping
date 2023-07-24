import time

import numpy as np
import rerun as rr

from mjregrasping.homotopy_utils import get_h_signature, discretize_path, skeleton_field_dir


def main():
    rr.init("homotopy_demo")
    rr.connect()
    # rr.save("homotopy_demo.rrd")

    rr.log_view_coordinates("world", up="+Z", timeless=True)
    rr.log_arrow('world_x', [0, 0, 0], [1, 0, 0], color=(255, 0, 0), width_scale=0.02, timeless=True)
    rr.log_arrow('world_y', [0, 0, 0], [0, 1, 0], color=(0, 255, 0), width_scale=0.02, timeless=True)
    rr.log_arrow('world_z', [0, 0, 0], [0, 0, 1], color=(0, 0, 255), width_scale=0.02, timeless=True)

    skeleton = np.array([
        [0.5, -2.5, -3.4],
        [0.5, -2.5, 1.8],
        [0.5, 0.5, 1.8],
        [0.5, 0.5, -3.4],
        [0.5, -2.5, -3.4],
    ])
    rr.log_line_strip(f'skeleton', skeleton, color=[0, 255, 0])

    # for ring_z in [0.2, 0.6, 1.2]:
    #     ring = make_ring(ring_z)
    #
    #     visualize_homotopy_for_path(ring, skeleton)

    # radius = 0.4
    # angles = np.arange(0,  np.pi, 0.2)
    # path1 = np.stack([radius * np.cos(angles), -0.1 + radius * np.sin(angles), np.linspace(0, 0.5, angles.shape[0])], -1)
    # visualize_homotopy_for_path(path1, skeleton)
    # path2 = np.stack([radius * np.cos(angles), 0.1 + -radius * np.sin(angles), np.linspace(0, 0.5, angles.shape[0])], -1)
    # visualize_homotopy_for_path(path2, skeleton)
    # path3 = np.array([
    #     [-0.5, 0.45, 0.2],
    #     [0.5, 0.45, 0.2],
    # ])
    # visualize_homotopy_for_path(path3, skeleton)
    # path3 = np.array([
    #     [-0.5, 0.55, 0.2],
    #     [0.5, 0.55, 0.2],
    # ])
    # visualize_homotopy_for_path(path3, skeleton)

    def _I(_path):
        path_discretized = discretize_path(_path)
        path_deltas = np.diff(path_discretized, axis=0)
        bs = skeleton_field_dir(skeleton, path_discretized[:-1])
        I = 0
        for b_i, p_i, delta_i in zip(bs, path_discretized, path_deltas):
            dI = np.dot(b_i, delta_i)
            I += dI
        return I

    points = []
    colors = []
    for x in np.arange(0, 1, 0.04):
        for y in np.arange(0, 1, 0.04):
            p = np.array([x, y, 0])
            points.append(p)
            I = _I(np.stack((np.zeros(3), p)))
            print(I)
            if I < 0:
                color = [0, -I, 0]
            else:
                color = [I, 0, 0]
            colors.append(color)
            rr.log_point(f"I/{x}/{y}", p, color=color, ext={'I': f'{I:.4f}'}, radius=0.01)


def visualize_homotopy_for_path(path, skeleton):
    path_discretized = discretize_path(path)
    path_deltas = np.diff(path_discretized, axis=0)
    bs = skeleton_field_dir(skeleton, path_discretized[:-1])
    h = get_h_signature(path, {'obs': skeleton})
    rr.log_line_strip("path", path, stroke_width=0.01, color=[255, 0, 0], ext={'h': str(h)})
    # integrate the field along the ring
    I = 0
    for b_i, p_i, delta_i in zip(bs, path_discretized, path_deltas):
        dI = np.dot(b_i, delta_i)
        rr.log_arrow('b', p_i, b_i / 2, width_scale=0.015)
        I += dI
        rr.log_scalar("I", I)

        time.sleep(0.003)
    h = abs(int(I.round(0)))
    print(f'{I:.4f}, {h}')


def make_ring(ring_z):
    radius = 0.25
    angles = np.arange(0, 2 * np.pi, 0.5)
    angles = np.append(angles, 0)
    ones = np.ones_like(angles)
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles), ones * ring_z], -1)
    return ring


if __name__ == '__main__':
    main()
