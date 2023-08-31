import time
from arc_utilities.listener import Listener

import weakref
from copy import deepcopy
from threading import Thread

import mujoco
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
        self.js_listener = Listener('/hdt_adroit_coms/joint_telem', JointState, queue_size=10)
        self.command_thread = Thread(target=RealValCommander.command_thread_func, args=(weakref.proxy(self),))
        self.command_thread.start()
        self.latest_cmd = JointState()
        for name in robot.joint_names:
            self.latest_cmd.name.append(name)

    def get_latest_joint_state(self) -> JointState:
        return self.js_listener.get()

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

    def send_vel_command(self, m: mujoco.MjModel, command):
        if not self.first_valid_command:
            self.first_valid_command = True

        # Compute the desired positions based on the joint limits, and the desired velocity direction
        low = m.actuator_actrange[:, 0]
        high = m.actuator_actrange[:, 1]
        positions = np.where(command >= 0, high, low)

        # Val expects duplicate positions and velocities for the grippers, which are at indices 9 and 17
        # and need to go to indices 10 and 19
        positions_dup = np.insert(positions, 10, positions[9])
        positions_dup = np.insert(positions_dup, 19, positions[17])
        command_dup = np.insert(command, 10, command[9])
        command_dup = np.insert(command_dup, 19, command[17])

        # Then update the latest command message with the new positions and velocities
        self.latest_cmd.header.stamp = rospy.Time.now()
        self.latest_cmd.position = positions_dup.tolist()
        self.latest_cmd.velocity = command_dup.tolist()

    def stop(self):
        self.should_disconnect = True
        self.command_thread.join()
