import mujoco
import rerun as rr

from mujoco import mjtSensor


class MjReRun:

    def __init__(self):
        pass

    def viz(self, m: mujoco.MjModel, d: mujoco.MjData):
        rr.set_time_seconds('sim_time', d.time)

        # 2D viz in rerun
        for sensor_idx in range(m.nsensor):
            sensor = m.sensor(sensor_idx)
            if sensor.type in [mjtSensor.mjSENS_TORQUE, mjtSensor.mjSENS_FORCE]:
                rr.log_scalar(f'sensor/{sensor.name}', float(d.sensordata[sensor.adr]))
        for joint_idx in range(m.njnt):
            joint = m.joint(joint_idx)
            if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                rr.log_scalar(f'qpos/{joint.name}', float(d.qpos[joint.qposadr]))

        rr.log_scalar(f'contact/num_contacts', len(d.contact))
