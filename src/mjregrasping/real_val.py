import time
# noinspection PyUnresolvedReferences
import tf2_geometry_msgs
import weakref
from copy import deepcopy
from threading import Thread

import mujoco
import numpy as np

import rospy
from arc_utilities.listener import Listener
from arc_utilities.tf2wrapper import TF2Wrapper
from arm_video_recorder.srv import TriggerVideoRecording, TriggerVideoRecordingRequest
from geometry_msgs.msg import PointStamped
from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.grasping import activate_grasp, get_grasp_locs
from mjregrasping.physics import Physics
from arm_robots_msgs.msg import GripperConstraint
from sensor_msgs.msg import JointState
from arm_robots_msgs.srv import SetCDCPDState, SetCDCPDStateRequest, SetGripperConstraints, SetGripperConstraintsRequest
from visualization_msgs.msg import MarkerArray


class RealValCommander:

    def __init__(self, phy: Physics):
        """
        All commands are ordered based on mujoco!
        """
        self.robot = phy.o.robot
        self.command_rate = rospy.Rate(100)
        self.should_disconnect = False
        self.first_valid_command = False
        self.command_pub = rospy.Publisher("/hdt_adroit_coms/joint_cmd", JointState, queue_size=10)
        self.js_listener = Listener('/hdt_adroit_coms/joint_telem', JointState, queue_size=10)
        self.command_thread = Thread(target=RealValCommander.command_thread_func, args=(weakref.proxy(self),))
        self.command_thread.start()
        self.latest_cmd = JointState()
        for name in self.robot.joint_names:
            self.latest_cmd.name.append(name)
        js = self.get_latest_joint_state()
        self.mj_order = []
        for mj_name in phy.o.robot.joint_names:
            self.mj_order.append(js.name.index(mj_name))

        self.cdcpd_sub = Listener("/cdcpd_pred", MarkerArray)
        self.set_cdcpd_srv = rospy.ServiceProxy("set_cdcpd_state", SetCDCPDState)
        self.set_cdcpd_grippers_srv = rospy.ServiceProxy("set_gripper_constraints", SetGripperConstraints)
        self.record_srv = rospy.ServiceProxy("video_recorder", TriggerVideoRecording)
        self.tfw = TF2Wrapper()

    def start_record(self):
        req = TriggerVideoRecordingRequest()
        now = int(time.time())
        req.filename = f'/home/peter/recordings/peter/real_untangle_{now}.avi'
        req.record = True
        req.timeout_in_sec = 10 * 60
        self.record_srv(req)

    def stop_record(self):
        req = TriggerVideoRecordingRequest()
        req.record = False
        self.record_srv(req)

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

    def send_vel_command(self, m: mujoco.MjModel, vel):
        if not self.first_valid_command:
            self.first_valid_command = True

        # Compute the desired positions based on the joint limits, and the desired velocity direction
        low = m.actuator_actrange[:, 0]
        high = m.actuator_actrange[:, 1]
        positions = np.where(vel >= 0, high, low)

        # Val expects duplicate positions and velocities for the grippers, which are at indices 9 and 17
        # and need to go to indices 10 and 19
        positions_dup = np.insert(positions, 10, positions[9])
        positions_dup = np.insert(positions_dup, 19, positions[17])
        command_dup = np.insert(vel, 10, vel[9])
        command_dup = np.insert(command_dup, 19, vel[17])

        # Then update the latest command message with the new positions and velocities
        self.latest_cmd.header.stamp = rospy.Time.now()
        self.latest_cmd.position = positions_dup.tolist()
        self.latest_cmd.velocity = command_dup.tolist()

    def send_pos_command(self, pos: np.ndarray):
        if not self.first_valid_command:
            self.first_valid_command = True

        self.latest_cmd.header.stamp = rospy.Time.now()
        self.latest_cmd.position = pos
        self.latest_cmd.velocity = []

    def stop(self):
        self.should_disconnect = True
        self.command_thread.join()

    def update_mujoco_qpos(self, phy: Physics):
        pos_in_mj_order = self.get_latest_qpos_in_mj_order()
        phy.d.qpos[phy.o.robot.qpos_indices] = pos_in_mj_order
        mujoco.mj_forward(phy.m, phy.d)

    def get_latest_qpos_in_mj_order(self):
        js = self.get_latest_joint_state()
        pos_in_mj_order = np.array(js.position)[self.mj_order]
        return pos_in_mj_order

    def pid_to_joint_configs(self, qs):
        for q in qs[:-1]:
            self.send_pos_command(q)
            self.wait_until_reached(q, reached_tol=4, stopped_tol=10)
        self.send_pos_command(qs[-1])
        self.wait_until_reached(qs[-1], reached_tol=2, stopped_tol=1)

    def wait_until_reached(self, q_target, reached_tol=2, stopped_tol=1):
        stopped = False
        reached = False
        for i in range(75):
            current_q = self.get_latest_qpos_in_mj_order()
            error = np.abs(current_q - q_target)
            max_joint_error = np.rad2deg(np.max(error))
            reached = max_joint_error < reached_tol
            stopped = np.rad2deg(np.max(np.abs(self.latest_cmd.velocity))) < stopped_tol
            if reached and stopped:
                return
        print(f"WARNING: reached timeout! {reached} {stopped}")

    def pull_rope_towards_cdcpd(self, phy: Physics, n_sub_time: int):
        cdcpd_pred = self.cdcpd_sub.get()
        # Now we have the cdcpd prediction, we need to tell mujoco to obey it
        # we do this by using eq constraints in mujoco. There is one eq constraint per rope body, but
        # cdcpd tracks the tips of the bodies, so we need to figure out which body each tip belongs to
        # and then activate the eq constraint for that capsule and step the simulation a little

        # first convert to numpy array [N, 3]
        cdcpd_n_points = len(cdcpd_pred.markers) - 1  # -1 for the linestrip marker
        for i, (marker, loc) in enumerate(zip(cdcpd_pred.markers, np.linspace(0, 1, cdcpd_n_points))):
            pos_in_cam = PointStamped()
            pos_in_cam.header.frame_id = 'zivid_optical_frame'
            pos_in_cam.point = marker.pose.position
            pos_in_world = self.tfw.transform_to_frame(pos_in_cam, target_frame='world')
            eq = phy.m.eq(f'B_{i}')
            eq.data[0:3] = np.array([pos_in_world.point.x, pos_in_world.point.y, pos_in_world.point.z])
            activate_grasp(phy, f'B_{i}', loc)

        phy.d.ctrl[:] = 0
        mujoco.mj_step(phy.m, phy.d, nstep=n_sub_time)

        # now deactivate all the eqs, so they don't stop the robot from moving
        for i in range(cdcpd_n_points):
            eq = phy.m.eq(f'B_{i}')
            eq.active = 0

    def set_cdcpd_from_mj_rope(self, phy: Physics):
        set_cdcpd_req = SetCDCPDStateRequest()
        for v in get_rope_points(phy):
            p_in_world = PointStamped()
            p_in_world.point.x = v[0]
            p_in_world.point.y = v[1]
            p_in_world.point.z = v[2]
            p_in_world.header.frame_id = "world"
            p_in_cam = self.tfw.transform_to_frame(p_in_world, target_frame='zivid_optical_frame')
            set_cdcpd_req.vertices.append(p_in_cam.point)
        self.set_cdcpd_srv(set_cdcpd_req)

    def set_cdcpd_grippers(self, phy: Physics):
        """ Set the CDCPD grasp constraints based on the current grasp locations. """
        req = SetGripperConstraintsRequest()
        locs = get_grasp_locs(phy)
        for eq_name, loc in zip(phy.o.rd.rope_grasp_eqs, locs):
            eq = phy.m.eq(eq_name)
            if eq.active:
                c = GripperConstraint()
                c.tf_frame_name = f'{eq_name}_tool'
                c.cdcpd_node_index = int(np.round(loc * 24))
                req.constraints.append(c)
        self.set_cdcpd_grippers_srv(req)
