#!/usr/bin/env python3
import logging
from time import sleep

import mujoco
import numpy as np
import rerun as rr
from mujoco import viewer

from mjregrasping.buffer import Buffer
from mjregrasping.rerun_visualizer import MjReRun

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

    joint_to_qv_map = make_joint_qv_map(m)
    mjrr = MjReRun(xml_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while True:
            viewer.sync()

            mjrr.viz(m, d)

            qpos_indices_for_act = np.array([m.actuator(i).actadr[0] for i in range(m.na)])
            qpos_for_act = d.qpos[qpos_indices_for_act]
            d.act = qpos_for_act + np.clip(d.act - qpos_for_act, -0.01, 0.01)

            mujoco.mj_step(m, d, nstep=10)
            sleep(0.05)


def make_joint_qv_map(model):
    # Iterate ove all the joints and get their DOFs
    # the total DOFs in all joints should equal model.nv
    my_nv = 0
    joint_to_qv_map = {}
    for i in range(model.njnt):
        joint = model.joint(i)
        if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
            joint_to_qv_map[joint.id] = joint.qposadr[0]
            joint_nv = 1
        elif joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
            joint_nv = 1
        elif joint.type == mujoco.mjtJoint.mjJNT_FREE:
            joint_nv = 6
        elif joint.type == mujoco.mjtJoint.mjJNT_BALL:
            joint_nv = 3
        else:
            raise NotImplementedError('Unsupported joint type')

        my_nv += joint_nv
        logger.debug(f'Joint {joint.name} has {joint_nv} DOFs')
        # only 1-dof hinge joints are supported
    logger.debug(f'Total DOFs: {my_nv} vs {model.nv}')
    return joint_to_qv_map


if __name__ == '__main__':
    main()
