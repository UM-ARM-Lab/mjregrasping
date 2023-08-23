import mujoco
import numpy as np

import rospy
from arc_utilities import ros_init
from mjregrasping.real_val import RealValCommander
from mjregrasping.robot_data import val as val_data


@ros_init.with_ros("fiddle")
def main():
    np.set_printoptions(precision=5, suppress=True, linewidth=200)

    m = mujoco.MjModel.from_xml_path("models/val.xml")
    j = m.actuator("joint6_vel").trnid[0]

    real_val = RealValCommander(val_data)

    t = 0
    v = 0.45
    while True:
        ctrl = np.zeros(m.nu)
        ctrl[j] = v
        real_val.send_vel_command(m, ctrl)
        if t % 1000 == 0:
            v = -v
        rospy.sleep(0.001)
        t += 1


if __name__ == '__main__':
    main()
