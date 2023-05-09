#!/usr/bin/env python3
from time import sleep

import mujoco
import numpy as np
import rerun as rr
from mujoco import viewer

from mjregrasping.buffer import Buffer


def main():
    np.set_printoptions(precision=3, suppress=True)
    rr.init('sim_viewer')
    rr.spawn()
    model = mujoco.MjModel.from_xml_path('models/test_ext.xml')
    data = mujoco.MjData(model)

    b1 = Buffer(10)
    b2 = Buffer(10)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while True:
            rr.set_time_seconds('sim_time', data.time)
            viewer.sync()

            torque1 = data.sensor('torque1').data
            torque2 = data.sensor('torque2').data
            torque1_dir = np.sign(np.dot(model.joint("joint1").axis, torque1))
            torque2_dir = np.sign(np.dot(model.joint("joint2").axis, torque2))
            torque1_mag_signed = torque1_dir * np.linalg.norm(torque1)
            torque2_mag_signed = torque2_dir * np.linalg.norm(torque2)
            external_torque1 = torque1_mag_signed - data.qfrc_bias[0]
            external_torque2 = torque2_mag_signed - data.qfrc_bias[1]

            b1.insert(external_torque1)
            b2.insert(external_torque2)

            external_torques = np.array([np.mean(b1.data), np.mean(b2.data)])
            if (b1.full() and b2.full()) and np.any(np.abs(external_torques) > 0.5):
                print("High torque!")
                data.act = data.qpos + np.clip(data.act - data.qpos, -0.005, 0.05)

            mujoco.mj_step(model, data, nstep=10)
            sleep(0.01)

            rr.log_scalar('scalars/torque/1', torque1_mag_signed)
            rr.log_scalar('scalars/torque/2', torque2_mag_signed)
            rr.log_scalar('scalars/qfrc_bias/1', data.qfrc_bias[0])
            rr.log_scalar('scalars/qfrc_bias/2', data.qfrc_bias[1])
            rr.log_scalar('scalars/external_torque/1', np.mean(b1.data))
            rr.log_scalar('scalars/external_torque/2', np.mean(b2.data))

            # override the controls set in the UI
            rr.log_scalar(f'controls/qpos/1', data.qpos[0])
            rr.log_scalar(f'controls/qpos/2', data.qpos[1])
            rr.log_scalar(f'controls/v/1', data.ctrl[0])
            rr.log_scalar(f'controls/v/2', data.ctrl[1])
            rr.log_scalar(f'controls/act/1', data.act[0])
            rr.log_scalar(f'controls/act/2', data.act[1])


if __name__ == '__main__':
    main()
