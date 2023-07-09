#!/usr/bin/env python
import argparse
import pathlib
import pickle
from time import perf_counter

import hjson
import mujoco
import numpy as np
import rerun as rr

from mjregrasping.mujoco_objects import Object
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.voxelgrid import VoxelGrid


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

    rr.init("mjcf_scene_to_vg")
    rr.connect()

    args = parser.parse_args()

    model_filename = args.model_filename
    res = args.res
    xmin = args.xmin
    xmax = args.xmax
    ymin = args.ymin
    ymax = args.ymax
    zmin = args.zmin
    zmax = args.zmax

    m = mujoco.MjModel.from_xml_path(args.model_filename.as_posix())
    d = mujoco.MjData(m)
    phy = Physics(m, d)
    mujoco.mj_forward(m, d)

    mjrr = MjReRun(model_filename)
    mjrr.viz(phy)

    nickname = model_filename.stem.removesuffix("_scene")
    outdir = model_filename.parent.parent / 'voxelgrids'
    outdir.mkdir(exist_ok=True)
    outfilename = outdir / (nickname + "_vg.pkl")
    args_filename = outdir / (nickname + "_args.txt")
    args_dict = {
        'model_filename': model_filename.as_posix(),
        'res':            res,
        'xmin':           xmin,
        'xmax':           xmax,
        'ymin':           ymin,
        'ymax':           ymax,
        'zmin':           zmin,
        'zmax':           zmax,
    }
    with args_filename.open("w") as args_f:
        hjson.dump(args_dict, args_f)
    origin_point = np.array([xmin, ymin, zmin]) + res / 2  # center  of the voxel [0,0,0]
    res = np.array([res])
    extent = np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])

    obstacle = Object(phy.m, 'computer_rack')

    t0 = perf_counter()
    vg = VoxelGrid(phy, res, extent, obstacle)
    print(f'Computing VoxelGrid: {perf_counter() - t0:.3f}')

    rr.log_points('point_cloud', vg.points)
    position = np.mean(extent, axis=1)
    half_size = (extent[:, 1] - extent[:, 0]) / 2
    rr.log_obb('extent', half_size=half_size, position=position)

    # TODO: sdf's can't be pickled at the moment so instead we save the voxel grid
    # sdf = compute_sdf(vg.vg, res, origin_point)

    print(f"Saving VoxelGrid to {outfilename}")
    with outfilename.open("wb") as f:
        pickle.dump(vg, f)


if __name__ == '__main__':
    main()
