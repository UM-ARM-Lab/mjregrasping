import time

import mujoco
import numpy as np
import rerun as rr
import hjson

from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun


def main():
    rr.init('viz_skeleton')
    rr.connect()

    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    np.seterr(all='raise')

    model_xml_prefix = 'computer_rack'
    model_xml_filename = f"models/{model_xml_prefix}_scene.xml"
    skeleton_filename = f"models/{model_xml_prefix}_skeleton.hjson"

    mjrr = MjReRun(model_xml_filename)

    m = mujoco.MjModel.from_xml_path(model_xml_filename)
    d = mujoco.MjData(m)
    phy = Physics(m, d)

    mujoco.mj_forward(m, d)

    while True:
        try:
            with open(skeleton_filename, 'r') as f:
                skeletons = hjson.load(f)
            for name, skeleton in skeletons.items():
                skeleton = np.array(skeleton)
                rr.log_line_strip(f'skeleton/{name}', skeleton, color=(0, 255, 0, 255), timeless=True)

            mjrr.viz(phy)
        except Exception as e:
            print(e)

        time.sleep(1)


if __name__ == '__main__':
    main()
