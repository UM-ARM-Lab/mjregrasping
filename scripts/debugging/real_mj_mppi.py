import multiprocessing
import time
import weakref
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from threading import Thread

import mujoco
import numpy as np
import rerun as rr

import rospy
from arc_utilities import ros_init
from mjregrasping.mujoco_objects import Objects, Object
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.regrasping_mppi import RegraspMPPI
from mjregrasping.robot_data import val
from mjregrasping.scenarios import val_untangle
from mjregrasping.viz import make_viz
from sensor_msgs.msg import JointState


class RealValCommander:

    def __init__(self, robot: Object):
        self.robot = robot
        self.command_rate = rospy.Rate(100)
        self.should_disconnect = False
        self.first_valid_command = False
        self.command_pub = rospy.Publisher("/hdt_adroit_coms/joint_cmd", JointState, queue_size=10)
        self.command_thread = Thread(target=RealValCommander.command_thread_func, args=(weakref.proxy(self),))
        self.command_thread.start()
        self.latest_cmd = JointState()
        for name in robot.joint_names:
            self.latest_cmd.name.append(name)

    def command_thread_func(self):
        try:
            while not self.first_valid_command:
                if self.should_disconnect:
                    break
                time.sleep(0.1)

            while True:
                if self.should_disconnect:
                    break
                # actually send commands periodically
                now = rospy.Time.now()
                time_since_last_command = now - self.latest_cmd.header.stamp
                if time_since_last_command < rospy.Duration(secs=1):
                    command_to_send = deepcopy(self.latest_cmd)
                    command_to_send.header.stamp = now
                    self.command_pub.publish(command_to_send)
                else:
                    rospy.logdebug_throttle(1, "latest command is too old, ignoring", logger_name="hdt_michigan")
                self.command_rate.sleep()
        except ReferenceError:
            pass

    def send_vel_command(self, phy, command):
        if not self.first_valid_command:
            self.first_valid_command = True

        # Compute the desired positions based on the joint limits, and the desired velocity direction
        low = phy.m.actuator_actrange[:, 0]
        high = phy.m.actuator_actrange[:, 1]
        positions = np.where(command >= 0, high, low)

        # Then update the latest command message with the new positions and velocities
        self.latest_cmd.header.stamp = rospy.Time.now()
        self.latest_cmd.position = positions.tolist()
        self.latest_cmd.velocity = command.tolist()


@ros_init.with_ros("real_mj_mppi")
def main():
    scenario = val_untangle
    goal_point = np.array([1.0, 0.0, 1.0])

    rr.init('real_mj_mppi')
    rr.connect()

    viz = make_viz(scenario)

    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
    d = mujoco.MjData(m)
    objects = Objects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
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
