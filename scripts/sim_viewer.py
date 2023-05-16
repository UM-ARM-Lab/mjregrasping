#!/usr/bin/env python3
import logging
from time import sleep

import mujoco
import numpy as np
import rerun as rr

from mjregrasping.buffer import Buffer
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rollout import limit_actuator_windup

logger = logging.getLogger(__name__)


class SensorBuffer(Buffer):

    def __init__(self, sensor_name, buffer_size):
        super().__init__(buffer_size)
        self.joint_name = sensor_name.replace("_torque", "")
        self.sensor_name = sensor_name


def main():
    np.set_printoptions(precision=3, suppress=True)
    rr.init('sim_viewer')
    rr.spawn()
    xml_path = 'models/untangle_scene.xml'
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    # iterate over all the sensors, check if they are torque sensors,
    # and if so, add a buffer to store the torque values
    buffers = []
    torque_sensors = []
    for i in range(m.nsensor):
        sensor = m.sensor(i)
        if sensor.type == mujoco.mjtSensor.mjSENS_TORQUE:
            buffers.append(SensorBuffer(sensor.name, 10))
            torque_sensors.append(sensor)

    mjrr = MjReRun(xml_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while True:
            viewer.sync()

            mjrr.viz(m, d)

            limit_actuator_windup(d, m)

            mujoco.mj_step(m, d, nstep=10)
            sleep(0.05)


if __name__ == '__main__':
    main()
