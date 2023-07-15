import mujoco
import numpy as np


def is_child(model, body, parent_body_idx):
    # traverse up the tree until, keeping a reference to "body"
    parent_idx = body.parentid
    while parent_idx != 0:
        if parent_idx == parent_body_idx:
            return body
        parent_idx = model.body(parent_idx).parentid
    return None


class MjObject:

    def __init__(self, m, parent_body_name):
        parent_body_idx = m.body(parent_body_name).id
        self.body_names = []
        self.geom_names = []
        self.joint_names = []
        self.body_indices = []
        self.geom_indices = []
        self.qpos_indices = []
        for joint_idx in range(m.njnt):
            joint = m.joint(joint_idx)
            body_id = m.jnt_bodyid[joint_idx]
            jnt_body = m.body(body_id)
            if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                dof = 1
            elif joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
                dof = 1
            elif joint.type == mujoco.mjtJoint.mjJNT_BALL:
                dof = 4  # orientation represented as quaternion
            elif joint.type == mujoco.mjtJoint.mjJNT_FREE:
                dof = 7
            else:
                raise NotImplementedError(f"Joint type {joint.type} not implemented")
            start = joint.qposadr[0]
            end = start + dof
            if is_child(m, jnt_body, parent_body_idx) or parent_body_idx == jnt_body.id:
                self.qpos_indices.extend(list(range(start, end)))
                self.joint_names.append(joint.name)

        for body_idx in range(m.nbody):
            body = m.body(body_idx)
            if body := is_child(m, body, parent_body_idx):
                self.body_indices.append(body.id)
                self.body_names.append(body.name)
                for geom_idx in range(int(body.geomadr), int(body.geomadr + body.geomnum)):
                    self.geom_indices.append(geom_idx)
                    self.geom_names.append(m.geom(geom_idx).name)

        self.body_indices = np.array(self.body_indices)
        self.geom_indices = np.array(self.geom_indices)
        self.qpos_indices = np.array(self.qpos_indices)
