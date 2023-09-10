import sys
import time
from dataclasses import dataclass
from enum import Enum, auto

import mujoco.viewer
import numpy as np
import rerun as rr
from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow, QApplication

import rospy
from mjregrasping.grasping import activate_grasp, let_rope_move_through_gripper_geoms
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.physics import Physics
from mjregrasping.rollout import limit_actuator_windup, slow_when_eqs_bad
from mjregrasping.scenarios import real_untangle
from mjregrasping.viz import make_viz


class CmdType(Enum):
    SAVE = auto()
    GRASP = auto()
    RELEASE = auto()
    START_RECORDING = auto()
    STOP_RECORDING = auto()
    DISABLE_IMS = auto()
    ENABLE_IMS = auto()
    RESET_TO_IMS = auto()


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
        self.reset_button.clicked.connect(self.reset_to_ims)
        self.show()

        self.latest_cmd = None
        self.save_filename = None
        self.loc = None
        self.eq_name = None

    def reset_to_ims(self):
        self.latest_cmd = CmdType.RESET_TO_IMS

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

    rr.init("viewer")
    rr.connect()

    scenario = real_untangle

    viz = make_viz(scenario)

    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
    d = mujoco.MjData(m)
    phy = Physics(m, d, MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))

    with mujoco.viewer.launch_passive(phy.m, phy.d) as viewer:
        with viewer.lock():
            viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

        app = QApplication(sys.argv)

        controls_window = InteractiveControls()
        now = int(time.time())

        dt = phy.m.opt.timestep * 10
        n_sub_time = int(dt / phy.m.opt.timestep)

        def _update_sim():
            latest_cmd = controls_window.latest_cmd

            step_start = time.time()

            viewer.sync()
            viz.viz(phy, is_planning=False)

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            elif latest_cmd == CmdType.GRASP:
                activate_grasp(phy, controls_window.eq_name, controls_window.loc)
                let_rope_move_through_gripper_geoms(phy)

            elif latest_cmd == CmdType.RELEASE:
                phy.m.eq(controls_window.eq_name).active = 0

            slow_when_eqs_bad(phy)
            limit_actuator_windup(phy)
            mujoco.mj_step(phy.m, phy.d, nstep=n_sub_time)

            # ensure events are processed only once
            controls_window.latest_cmd = None

        timer = QTimer()
        timer.timeout.connect(_update_sim)
        sim_step_ms = int(phy.m.opt.timestep * 1000)
        timer.start(sim_step_ms)

        sys.exit(app.exec())


if __name__ == '__main__':
    main()
