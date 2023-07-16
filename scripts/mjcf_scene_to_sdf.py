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
from time import perf_counter

import hjson
import mujoco
import numpy as np
import pysdf_tools
import rerun as rr

from mjregrasping.mujoco_object import MjObject
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.voxelgrid import make_vg, get_points_and_values


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

    position = np.mean(extent, axis=1)
    half_size = (extent[:, 1] - extent[:, 0]) / 2
    rr.log_obb('extent', half_size=half_size, position=position)

    t0 = perf_counter()
    vg, points = make_vg(phy, res, extent, origin_point)
    print(f'Computing VoxelGrid: {perf_counter() - t0:.3f}')

    oob_value = pysdf_tools.COLLISION_CELL(-10000)
    sdf_result = vg.ExtractSignedDistanceField(oob_value.occupancy, False, False)
    sdf: pysdf_tools.SignedDistanceField = sdf_result[0]
    print(f"Saving to {outfilename}")
    sdf.SaveToFile(str(outfilename), True)

    # Visualize slice sof the SDF in rerun:
    red = np.array([1, 0, 0, 1.0])
    green = np.array([0, 1, 0, 1.0])
    slice_axis = 2
    sdf_dims = [sdf.GetNumXCells(), sdf.GetNumYCells(), sdf.GetNumZCells()]

    def x_slices(i, axis):
        n_j = sdf_dims[(axis + 1) % 3]
        n_k = sdf_dims[(axis + 2) % 3]
        for j, k in np.ndindex(n_j, n_k):
            if axis == 0:
                yield i, j, k
            elif axis == 1:
                yield k, i, j
            elif axis == 2:
                yield j, k, i
            else:
                raise ValueError(f"Invalid axis {axis}, must be 0, 1 or 2")

    for slice_i in range(0, sdf_dims[slice_axis], 1):
        indices = x_slices(slice_i, slice_axis)
        points, values = get_points_and_values(sdf, indices)
        colors = [red if v < 0 else green for v in values]
        rr.log_points(f'sdf', positions=points, colors=colors, radii=sdf.GetResolution() / 2)


if __name__ == '__main__':
    main()
