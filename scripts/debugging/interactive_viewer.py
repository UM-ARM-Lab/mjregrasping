import sys
import time
from dataclasses import dataclass
from enum import Enum, auto

import rerun as rr
from pathlib import Path

import mujoco.viewer
from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow, QApplication

import rospy
from mjregrasping.grasping import activate_grasp
from mjregrasping.mjsaver import save_data_and_eq, load_data_and_eq
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.physics import Physics
from mjregrasping.rollout import limit_actuator_windup, slow_when_eqs_bad
from mjregrasping.scenarios import conq_hose, setup_conq_hose, cable_harness, setup_cable_harness, val_untangle, \
    setup_untangle
from mjregrasping.viz import make_viz


class CmdType(Enum):
    SAVE = auto()
    GRASP = auto()
    RELEASE = auto()


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
        self.release_button.clicked.connect(self.release)
        self.grasp_button.clicked.connect(self.grasp)
        self.show()

        self.latest_cmd = None
        self.save_filename = None
        self.loc = None
        self.eq_name = None

    def save(self):
        self.latest_cmd = CmdType.SAVE
        self.save_filename = self.save_filename_edit.text()

    def grasp(self):
        self.latest_cmd = CmdType.GRASP
        self.eq_name = self.grasp_name.text()
        self.loc = self.grasp_slider.value() / 100.0

    def release(self):
        self.latest_cmd = CmdType.RELEASE
        self.eq_name = self.release_name.text()
        self.loc = None


def main():
    rospy.init_node("interactive_viewer")
    scenario = cable_harness
    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))

    rr.init("viewer")
    rr.connect()

    d = mujoco.MjData(m)
    # state_path = Path("states/CableHarness/1689602983.pkl")
    # d = load_data_and_eq(m, state_path, True)
    phy = Physics(m, d, objects=MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))
    viz = make_viz(scenario)

    setup_cable_harness(phy, viz)

    root = Path(f"states/{scenario.name}")
    root.mkdir(exist_ok=True, parents=True)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        with viewer.lock():
            viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

        app = QApplication(sys.argv)

        controls_window = InteractiveControls()

        def _update_sim():
            latest_cmd = controls_window.latest_cmd

            step_start = time.time()

            viewer.sync()

            slow_when_eqs_bad(phy)
            limit_actuator_windup(phy)
            mujoco.mj_step(m, d)

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            if latest_cmd == CmdType.SAVE:
                now = int(time.time())
                path = root / f"{controls_window.save_filename}.pkl"
                print(f"Saving to {path}")
                save_data_and_eq(phy, path)
            elif latest_cmd == CmdType.GRASP:
                activate_grasp(phy, controls_window.eq_name, controls_window.loc)
            elif latest_cmd == CmdType.RELEASE:
                phy.m.eq(controls_window.eq_name).active = 0

            # ensure events are processed only once
            controls_window.latest_cmd = None

        timer = QTimer()
        timer.timeout.connect(_update_sim)
        sim_step_ms = int(m.opt.timestep * 100)
        timer.start(sim_step_ms)

        sys.exit(app.exec())


if __name__ == '__main__':
    main()
