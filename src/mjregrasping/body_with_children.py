from copy import copy

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


class Children:
    def __init__(self, model, parent_body_idx):
        self.body_names = []
        self.geom_names = []
        self.body_indices = []
        self.geom_indices = []
        self.qpos_indices = []
        for joint_idx in range(model.njnt):
            joint = model.joint(joint_idx)
            body_id = model.jnt_bodyid[joint_idx]
            jnt_body = model.body(body_id)
            if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                dof = 1
            elif joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
                dof = 1
            elif joint.type == mujoco.mjtJoint.mjJNT_BALL:
                dof = 4  # orientation represented as quaternion
            elif joint.type == mujoco.mjtJoint.mjJNT_FREE:
                dof = 6
            start = joint.qposadr[0]
            end = start + dof
            if is_child(model, jnt_body, parent_body_idx):
                self.qpos_indices.extend(list(range(start, end)))

        for body_idx in range(model.nbody):
            body = model.body(body_idx)
            if body := is_child(model, body, parent_body_idx):
                self.body_indices.append(body.id)
                self.body_names.append(body.name)
                for geom_idx in range(int(body.geomadr), int(body.geomadr + body.geomnum)):
                    self.geom_indices.append(geom_idx)
                    self.geom_names.append(model.geom(geom_idx).name)


class Object(Children):

    def __init__(self, model, parent_body_name):
        self.parent_body_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, parent_body_name)
        if self.parent_body_idx == -1:
            raise ValueError(f"body {parent_body_name} not found")
        Children.__init__(self, model, self.parent_body_idx)


class Objects:

    def __init__(self, model, obstacle_name='computer_rack'):
        self.val = Object(model, "val_base")
        self.rope = Object(model, "rope")
        self.obstacle = Object(model, obstacle_name)

        # allow collision between the environment and the finger pads
        self.val_collision_geom_names = copy(self.val.geom_names)
        self.val_collision_geom_names.remove('left_finger_pad')
        self.val_collision_geom_names.remove('left_finger_pad2')
        self.val_collision_geom_names.remove('right_finger_pad')
        self.val_collision_geom_names.remove('right_finger_pad2')

        self.val_self_collision_geom_names = copy(self.val.geom_names)
        self.val_gripper_act_names = [
            'leftgripper_vel',
            'leftgripper2_vel',
            'rightgripper_vel',
            'rightgripper2_vel',
        ]
        self.gripper_ctrl_indices = np.concatenate([model.actuator(n).actadr for n in self.val_gripper_act_names])
        self.val_self_collision_geom_names.remove('left_finger_pad')
        self.val_self_collision_geom_names.remove('left_finger_pad2')
        self.val_self_collision_geom_names.remove('right_finger_pad')
        self.val_self_collision_geom_names.remove('right_finger_pad2')
        self.val_self_collision_geom_names.remove('leftgripper')
        self.val_self_collision_geom_names.remove('leftgripper2')
        self.val_self_collision_geom_names.remove('rightgripper')
        self.val_self_collision_geom_names.remove('rightgripper2')
