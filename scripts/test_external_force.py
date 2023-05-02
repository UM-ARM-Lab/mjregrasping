import mujoco
import rerun as rr

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.mujoco_visualizer import MujocoVisualizer


def main():
    rospy.init_node('test_ext')

    rr.init('test_external_force')
    rr.connect()

    model = mujoco.MjModel.from_xml_path('models/test_ext.xml')
    data = mujoco.MjData(model)

    tfw = TF2Wrapper()
    mjviz = MujocoVisualizer(tfw)

    mujoco.mj_forward(model, data)
    mjviz.viz(model, data)

    data.ctrl[0] = 0.
    for t in range(1000):
        mujoco.mj_step(model, data, nstep=1)
        mjviz.viz(model, data)
        rospy.sleep(0.001)

    data.ctrl[0] = -0.05
    for t in range(1000):
        mujoco.mj_step(model, data, nstep=1)
        mjviz.viz(model, data)
        rospy.sleep(0.001)

    model.joint("slider").qpos_spring[0] = 0.15
    for t in range(3000):
        mujoco.mj_step(model, data, nstep=1)
        mjviz.viz(model, data)
        rospy.sleep(0.001)


if __name__ == '__main__':
    main()
