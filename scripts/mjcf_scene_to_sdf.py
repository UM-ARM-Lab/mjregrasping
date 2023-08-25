#!/usr/bin/env python
# This file depends on the following external libraries:
#  - sdf_tools python bindings
#  - mujoco
#  - hjson
# And the following files from within this project:
#  - mjregrasping/mujoco_object.py
#  - mjregrasping/physics.py
#  - mjregrasping/rerun_visualizer.py
#  - mjregrasping/voxelgrid.py
# You can comment out the rerun code if you don't want it installed,
# but then you won't be able to visualize anything,
# so it will hard to know if something isn't right.

import argparse
import pathlib

import rerun as rr

from arc_utilities import ros_init
from mjregrasping.mjcf_scene_to_sdf import mjcf_scene_to_sdf


@ros_init.with_ros("mjcf_scene_to_sdf")
def main():
    parser = argparse.ArgumentParser("Convert a mjcf scene to a voxel grid and save it as a pickle file.")
    parser.add_argument('model_filename', type=pathlib.Path)
    parser.add_argument('res', type=float)
    parser.add_argument('xmin', type=float)
    parser.add_argument('xmax', type=float)
    parser.add_argument('ymin', type=float)
    parser.add_argument('ymax', type=float)
    parser.add_argument('zmin', type=float)
    parser.add_argument('zmax', type=float)

    rr.init("mjcf_scene_to_sdf")
    rr.connect()

    args = parser.parse_args()

    mjcf_scene_to_sdf(args.model_filename, args.res,
                      args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax)


if __name__ == '__main__':
    main()
