import numpy as np
import rerun as rr
from multiset import Multiset

from mjregrasping.homotopy_utils import get_h_signature


def main():
    # Initialize rerun, the visualizer
    rr.init("homotopy_demo")
    rr.spawn()
    rr.log('world', rr.Transform3D())

    # create two loops of points that represent obstacles
    obstacle_loop1 = np.array([
        [-0.5, 0, -0.5],
        [0.5, 0, -0.5],
        [0.5, 0, 0.5],
        [-0.5, 0, 0.5],
        [-0.5, 0, -0.5],
    ])
    obstacle_loop2 = np.array([
        [-0.5, 0.75, -0.5],
        [0.5, 0.75, -0.5],
        [0.5, 0.75, 0.5],
        [-0.5, 0.75, 0.5],
        [-0.5, 0.75, -0.5],
    ])

    # log the obstacles to the visualizer
    rr.log('obstacle_loop1', rr.LineStrips3D(obstacle_loop1, colors=[0, 255, 0]))
    rr.log('obstacle_loop2', rr.LineStrips3D(obstacle_loop2, colors=[0, 255, 0]))

    # create a grasp loop, formed by the robot and cable
    grasp_loop = np.array([
        [0.0, -0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.5, 1.0],
        [0.0, -0.5, 1.0],
        [0.0, -0.5, 0.0],
    ])

    # log the grasp loop to the visualizer
    rr.log('grasp_loop', rr.LineStrips3D(grasp_loop, colors=[255, 0, 0]))

    # compute the GL-signature of the grasp loop
    skeletons = {
        'obs1': obstacle_loop1,
        'obs2': obstacle_loop2
    }

    gl_signature = Multiset([get_h_signature(grasp_loop, skeletons)])
    print(f"GL-signature: {gl_signature}")


if __name__ == '__main__':
    main()
