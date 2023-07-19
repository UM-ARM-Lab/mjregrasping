import time
import weakref
from copy import deepcopy
from threading import Thread

import numpy as np

import rospy
from mjregrasping.mujoco_object import MjObject
from sensor_msgs.msg import JointState


class RealValCommander:

    def __init__(self, robot: MjObject):
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
