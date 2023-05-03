import mujoco
import rerun as rr

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.mujoco_visualizer import MujocoVisualizer
from mujoco import viewer

def main():
    rospy.init_node('test_ext')

    rr.init('test_external_force')
    rr.connect()

    model = mujoco.MjModel.from_xml_path('models/test_ext.xml')
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = 1

        tfw = TF2Wrapper()
        mjviz = MujocoVisualizer(tfw)

        mujoco.mj_forward(model, data)
        mjviz.viz(model, data)

        while True:
            mujoco.mj_step(model, data, nstep=1)
            viewer.sync()
            mjviz.viz(model, data)
            rospy.sleep(0.001)


if __name__ == '__main__':
    main()
