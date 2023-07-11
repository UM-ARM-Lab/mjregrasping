#!/usr/bin/env python
import argparse
import pathlib
from time import perf_counter

import hjson
import mujoco
import numpy as np
import pysdf_tools
import rerun as rr

from mjregrasping.mujoco_objects import Object
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.voxelgrid import make_vg


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

    # Check that the cc sphere radius is half the resolution, otherwise print a warning
    cc_sphere_radius = m.geom('cc_sphere').size[0]
    if not np.isclose(cc_sphere_radius, res / 2):
        print(f"Warning: {cc_sphere_radius=} is not half the resolution ({res=})!")

    mjrr = MjReRun(model_filename)
    mjrr.viz(phy)

    nickname = model_filename.stem
    if nickname.endswith("_scene"):
        nickname = nickname[:-6]
    outdir = model_filename.parent.parent / 'sdfs'
    outdir.mkdir(exist_ok=True)
    outfilename = outdir / (nickname + ".sdf")
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

    position = np.mean(extent, axis=1)
    half_size = (extent[:, 1] - extent[:, 0]) / 2
    rr.log_obb('extent', half_size=half_size, position=position)

    t0 = perf_counter()
    vg, points = make_vg(phy, res, extent, origin_point, obstacle)
    print(f'Computing VoxelGrid: {perf_counter() - t0:.3f}')

    rr.log_points('point_cloud', points)

    print(f"Saving to {outfilename}")
    oob_value = pysdf_tools.COLLISION_CELL(-10000)
    sdf_result = vg.ExtractSignedDistanceField(oob_value.occupancy, False, False)
    sdf: pysdf_tools.SignedDistanceField = sdf_result[0]
    sdf.SaveToFile(str(outfilename), True)


if __name__ == '__main__':
    main()
