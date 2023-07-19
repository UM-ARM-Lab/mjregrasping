import multiprocessing
import time
from concurrent.futures.thread import ThreadPoolExecutor

import mujoco
import numpy as np
import rerun as rr

import rospy
from arc_utilities import ros_init
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.real_val import RealValCommander
from mjregrasping.regrasping_mppi import RegraspMPPI
from mjregrasping.robot_data import val
from mjregrasping.scenarios import val_untangle
from mjregrasping.viz import make_viz


@ros_init.with_ros("real_mj_mppi")
def main():
    scenario = val_untangle
    goal_point = np.array([1.0, 0.0, 1.0])

    rr.init('real_mj_mppi')
    rr.connect()

    viz = make_viz(scenario)

    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
    d = mujoco.MjData(m)
    objects = MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
    phy = Physics(m, d, objects)

    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)

    real_val = RealValCommander(phy.o.robot)

    with ThreadPoolExecutor(multiprocessing.cpu_count() - 1) as pool:
        mppi = RegraspMPPI(pool=pool, nu=phy.m.nu, seed=0, horizon=hp['regrasp_horizon'], noise_sigma=val.noise_sigma,
                           temp=hp['regrasp_temp'])
        for _ in range(30):
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown")

            # regrasp_goal.viz_goal(phy)
            # if regrasp_goal.satisfied(phy):
            #     print("Goal reached!")
            #     return True
            #
            # while warmstart_count < hp['warmstart']:
            #     command, sub_time_s = mppi.command(phy, regrasp_goal, num_samples, viz=viz)
            #     mppi_viz(mppi, regrasp_goal, phy, command, sub_time_s)
            #     warmstart_count += 1
            #
            # command, sub_time_s = mppi.command(phy, regrasp_goal, num_samples, viz=viz)
            # mppi_viz(mppi, regrasp_goal, phy, command, sub_time_s)
            #
            # control_step(phy, command, sub_time_s)
            viz.viz(phy)

            command = np.zeros(phy.m.nu)
            command[8] = -0.2
            time.sleep(0.5)

            real_val.send_vel_command(phy, command)


if __name__ == '__main__':
    main()
