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

            # iterate over each sensor and compute external torque
            external_torques = []
            for buffer in buffers:
                sensor = m.sensor(buffer.sensor_name)
                torque_rpy = d.sensor(sensor.name).data
                # convention that torque sensors start with the name of the joint they are attached to
                joint = m.joint(buffer.joint_name)
                qv_idx = joint_to_qv_map[joint.id]
                current_torque = joint.axis @ torque_rpy
                buffer.insert(current_torque)
                torque = np.mean(buffer.data)
                external_torque = torque - d.qfrc_bias[qv_idx]
                external_torques.append(external_torque)
                rr.log_scalar(f'qfrc_bias/{joint.name}', d.qfrc_bias[qv_idx])
                rr.log_scalar(f'external_torque/{joint.name}', external_torque)

            # FIXME: buffer is for the wrong variable
            external_torques = np.array(external_torques)
            if all([b.full() for b in buffers]) and np.any(np.abs(external_torques) > 3.0):
                print("High torque!")
                qpos_indices_for_sensor = np.array([m.joint(b.joint_name).qposadr[0] for b in buffers])
                qpos_for_sensors = d.qpos[qpos_indices_for_sensor]
                d.act = qpos_for_sensors + np.clip(d.act - qpos_for_sensors, -0.01, 0.01)

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
