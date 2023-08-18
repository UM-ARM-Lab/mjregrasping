import sys
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import mujoco.viewer
import numpy as np
import rerun as rr
from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow, QApplication
from scipy.linalg import block_diag

import rospy
from geometry_msgs.msg import Pose, Quaternion
from mjregrasping.basic_3d_pose_marker import Basic3DPoseInteractiveMarker
from mjregrasping.grasping import activate_grasp
from mjregrasping.jacobian_ctrl import get_w_in_tool, warn_near_joint_limits
from mjregrasping.mjsaver import save_data_and_eq, load_data_and_eq
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.my_transforms import xyzw_quat_from_matrix, xyzw_quat_to_matrix
from mjregrasping.physics import Physics, get_q
from mjregrasping.rollout import control_step, limit_actuator_windup, slow_when_eqs_bad
from mjregrasping.scenarios import cable_harness, setup_cable_harness
from mjregrasping.viz import make_viz
from ros_numpy import numpify, msgify


class CmdType(Enum):
    SAVE = auto()
    GRASP = auto()
    RELEASE = auto()
    START_RECORDING = auto()
    STOP_RECORDING = auto()
    DISABLE_IMS = auto()
    ENABLE_IMS = auto()


@dataclass
class Grasp:
    name: str
    loc: float


@dataclass
class Release:
    name: str


class InteractiveControls(QMainWindow):
    def __init__(self):
        super(InteractiveControls, self).__init__()
        uic.loadUi('interactive_viewer.ui', self)
        self.save_button.clicked.connect(self.save)
        self.grasp_left_button.clicked.connect(self.grasp_left)
        self.grasp_right_button.clicked.connect(self.grasp_right)
        self.release_left_button.clicked.connect(self.release_left)
        self.release_right_button.clicked.connect(self.release_right)
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.ims_enabled_checkbox.stateChanged.connect(self.ims_enabled_changed)
        self.show()

        self.latest_cmd = None
        self.save_filename = None
        self.loc = None
        self.eq_name = None

    def ims_enabled_changed(self):
        if self.ims_enabled_checkbox.isChecked():
            self.latest_cmd = CmdType.ENABLE_IMS
        else:
            self.latest_cmd = CmdType.DISABLE_IMS

    def start_recording(self):
        self.latest_cmd = CmdType.START_RECORDING

    def stop_recording(self):
        self.latest_cmd = CmdType.STOP_RECORDING

    def save(self):
        self.latest_cmd = CmdType.SAVE
        self.save_filename = self.save_filename_edit.text()

    def grasp_left(self):
        self.latest_cmd = CmdType.GRASP
        self.eq_name = 'left'
        self.loc = self.grasp_left_slider.value() / 100.0

    def grasp_right(self):
        self.latest_cmd = CmdType.GRASP
        self.eq_name = 'right'
        self.loc = self.grasp_right_slider.value() / 100.0

    def release_left(self):
        self.latest_cmd = CmdType.RELEASE
        self.eq_name = 'left'
        self.loc = None

    def release_right(self):
        self.latest_cmd = CmdType.RELEASE
        self.eq_name = 'right'
        self.loc = None


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rospy.init_node("interactive_viewer")
    scenario = cable_harness
    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))

    rr.init("viewer")
    rr.connect()

    viz = make_viz(scenario)
    d = mujoco.MjData(m)
    state_path = Path("states/CableHarness/init0.pkl")
    d = load_data_and_eq(m, state_path, True)
    phy = Physics(m, d, objects=MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))
    # setup_cable_harness(phy, viz)


    left_im = Basic3DPoseInteractiveMarker(name='left', scale=0.1)
    init_im_pose(left_im, phy, 'left_tool')
    right_im = Basic3DPoseInteractiveMarker(name='right', scale=0.1)
    init_im_pose(right_im, phy, 'right_tool')

    root = Path(f"states/{scenario.name}")
    root.mkdir(exist_ok=True, parents=True)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        with viewer.lock():
            viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

        app = QApplication(sys.argv)

        controls_window = InteractiveControls()
        is_recording = False
        ims_enabled = False
        now = int(time.time())
        recording_path = Path(f"demos/{scenario.name}/{now}")
        recording_path.mkdir(exist_ok=True, parents=True)

        dt = phy.m.opt.timestep * 10
        n_sub_time = int(dt / m.opt.timestep)

        def _update_sim():
            nonlocal is_recording, ims_enabled
            latest_cmd = controls_window.latest_cmd

            step_start = time.time()

            viewer.sync()
            viz.viz(phy, is_planning=False)

            left_twists_in_tool = get_twist_for_tool(phy, left_im, 'left_tool', viz)
            right_twists_in_tool = get_twist_for_tool(phy, right_im, 'right_tool', viz)
            twists_in_tool = np.concatenate([left_twists_in_tool, right_twists_in_tool])
            # set the initial ctrl based on the IMs
            ctrl = get_val_dual_jac_ctrl(phy, twists_in_tool)

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            gripper_vel = 0.1
            grasp_n_steps = 25
            if latest_cmd == CmdType.SAVE:
                path = root / f"{controls_window.save_filename}.pkl"
                print(f"Saving to {path}")
                save_data_and_eq(phy, path)
            elif latest_cmd == CmdType.GRASP:
                activate_grasp(phy, controls_window.eq_name, controls_window.loc)
                if 'left' in controls_window.eq_name:
                    ctrl[phy.m.actuator('leftgripper_vel').id] = -gripper_vel
                if 'right' in controls_window.eq_name:
                    ctrl[phy.m.actuator('rightgripper_vel').id] = -gripper_vel
                for _ in range(grasp_n_steps):
                    control_step(phy, ctrl, dt)
                    viewer.sync()
                    viz.viz(phy, is_planning=False)
            elif latest_cmd == CmdType.RELEASE:
                phy.m.eq(controls_window.eq_name).active = 0
                if 'left' in controls_window.eq_name:
                    ctrl[phy.m.actuator('leftgripper_vel').id] = gripper_vel
                if 'right' in controls_window.eq_name:
                    ctrl[phy.m.actuator('rightgripper_vel').id] = gripper_vel
                for _ in range(grasp_n_steps):
                    control_step(phy, ctrl, dt)
                    viewer.sync()
                    viz.viz(phy, is_planning=False)
            elif latest_cmd == CmdType.START_RECORDING:
                is_recording = True
            elif latest_cmd == CmdType.STOP_RECORDING:
                is_recording = False
            elif latest_cmd == CmdType.DISABLE_IMS:
                ims_enabled = False
            elif latest_cmd == CmdType.ENABLE_IMS:
                ims_enabled = True

            if is_recording:
                if int(d.time * 10) % 10 == 0:
                    now = int(time.time())
                    path = recording_path / f"{now}.pkl"
                    save_data_and_eq(phy, path)

            # Now step the simulation
            if ims_enabled:
                d.ctrl = ctrl
            slow_when_eqs_bad(phy)
            limit_actuator_windup(phy)
            mujoco.mj_step(m, d, nstep=n_sub_time)

            # ensure events are processed only once
            controls_window.latest_cmd = None

        timer = QTimer()
        timer.timeout.connect(_update_sim)
        sim_step_ms = int(m.opt.timestep * 1000)
        timer.start(sim_step_ms)

        sys.exit(app.exec())


def get_twist_for_tool(phy, im, tool_site_name, viz, w_scale=0.2):
    pose = im.get_pose()
    desired_position = numpify(pose.position)
    current_position = phy.d.site(tool_site_name).xpos
    kPos = 1
    v_in_world = (desired_position - current_position) * kPos
    viz.arrow(f'{tool_site_name}_v', current_position, v_in_world, color='m')
    tool2world_mat = phy.d.site(tool_site_name).xmat.reshape(3, 3)
    v_in_tool = tool2world_mat.T @ v_in_world
    desired_q_in_world = numpify(pose.orientation)
    desired_mat_in_world = xyzw_quat_to_matrix(desired_q_in_world)
    desired_mat_in_tool = tool2world_mat.T @ desired_mat_in_world
    w_in_tool = get_w_in_tool(desired_mat_in_tool, w_scale)
    twist_in_tool = np.concatenate([v_in_tool, w_in_tool])
    return twist_in_tool


def get_val_dual_jac_ctrl(phy: Physics, twists, jnt_lim_avoidance=0.25):
    left_J_gripper = get_site_jac(phy, 'left_tool')
    right_J_gripper = get_site_jac(phy, 'right_tool')
    J_gripper = np.concatenate([left_J_gripper, right_J_gripper])
    J_pinv = np.linalg.pinv(J_gripper)
    # use null-space projection to avoid joint limits
    current_q = get_q(phy)
    zero_vels = (phy.o.rd.q_home - current_q) * jnt_lim_avoidance
    warn_near_joint_limits(current_q, phy)
    ctrl = J_pinv @ twists + (np.eye(phy.m.nu) - J_pinv @ J_gripper) @ zero_vels
    return ctrl


def get_site_jac(phy, tool_site_name):
    tool_site = phy.m.site(tool_site_name)
    Jp = np.zeros((3, phy.m.nv))
    Jr = np.zeros((3, phy.m.nv))
    mujoco.mj_jacSite(phy.m, phy.d, Jp, Jr, tool_site.id)
    J_base = np.concatenate((Jp, Jr), axis=0)
    J_base = J_base[:, phy.m.actuator_trnid[:, 0]]
    # Transform J from base from to gripper frame
    tool2world_mat = phy.d.site_xmat[tool_site.id].reshape(3, 3)
    J_gripper = block_diag(tool2world_mat.T, tool2world_mat.T) @ J_base
    return J_gripper


def init_im_pose(im, phy, tool_site_name):
    current_pose = Pose()
    current_position = phy.d.site(tool_site_name).xpos
    current_mat = phy.d.site(tool_site_name).xmat.reshape(3, 3)
    current_pose.position.x = current_position[0]
    current_pose.position.y = current_position[1]
    current_pose.position.z = current_position[2]
    current_mat_full = np.eye(4)
    current_mat_full[:3, :3] = current_mat
    current_pose.orientation = msgify(Quaternion, xyzw_quat_from_matrix(current_mat_full))
    im.set_pose(current_pose)


if __name__ == '__main__':
    main()
