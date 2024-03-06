from copy import deepcopy
from typing import Optional

import mujoco
import numpy as np
from mujoco import mj_id2name, mju_str2Type

from mjregrasping.mujoco_objects import MjObjects


class Physics:
    # m: mujoco.MjModel
    # d: mujoco.MjData
    # o: MjObjects

    # def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, objects: Optional[MjObjects] = None):
    def __init__(self, physics):
        # self.m = m
        # self.d = d
        # self.o = objects
        self.p = physics
        self.d = physics.data
        self.m = physics.model
        # self.o = MJObjects(self.m, 'obj1', )

    def __copy__(self):
        raise NotImplementedError("Use .copy_data() or .copy_all() to avoid ambiguity")

    def copy_data(self):
        new_phy = Physics(deepcopy(self.p))
        # contact state is not copied, so we need to run forward to update it

        new_phy.p.step()
        return new_phy

    def copy_all(self):
        """ Much slower, since copying the model is slow """
        new_phy = Physics(deepcopy(self.p))
        # contact state is not copied, so we need to run forward to update it
        new_phy.p.step()
        return new_phy


def get_contact_forces(phy: Physics):
    contact_geoms1 = phy.d.contact.geom1
    contact_geoms2 = phy.d.contact.geom2

    contact_geoms = np.concatenate([contact_geoms1, contact_geoms2])
    contact_bodies = phy.m.geom_bodyid[contact_geoms]
    contact_forces = phy.d.cfrc_ext[contact_bodies]
    return contact_forces


def get_total_contact_force(phy: Physics):
    return np.sum(np.linalg.norm(get_contact_forces(phy), axis=-1))


def get_q(phy: Physics):
    """ only gets the q for actuated joints"""
    qpos_ids_for_actuators = get_qpos_ids_for_actuators(phy)
    return deepcopy(phy.d.qpos[qpos_ids_for_actuators])


def get_qpos_for_actuators(phy):
    ids = get_qpos_ids_for_actuators(phy)
    return phy.d.qpos[ids]


def get_qpos_for_actuator(phy, actuator_name: str):
    idx = get_qpos_id_for_actuator(phy, actuator_name)
    return phy.d.qpos[idx]


def get_qpos_ids_for_actuators(phy):
    joint_ids_for_actuators = phy.m.actuator_trnid[:, 0]
    qpos_ids_for_actuators = phy.m.jnt_qposadr[joint_ids_for_actuators]
    return qpos_ids_for_actuators


def get_qpos_id_for_actuator(phy, actuator_name: str):
    joint_id_for_actuator = phy.m.actuator(actuator_name).trnid[0]
    qpos_id_for_actuator = phy.m.jnt_qposadr[joint_id_for_actuator]
    return qpos_id_for_actuator


def get_full_q(phy: Physics):
    return deepcopy(phy.d.qpos[phy.o.robot.qpos_indices])


def get_parent_child_names(geom_bodyid: int, m: mujoco.MjModel):
    parent_bodyid = geom_bodyid
    body_name = mj_id2name(m, mju_str2Type("body"), geom_bodyid)
    child_name = body_name.split("/")[0]
    parent_name = child_name
    while True:
        parent_bodyid = m.body_parentid[parent_bodyid]
        _parent_name = mj_id2name(m, mju_str2Type("body"), parent_bodyid)
        if parent_bodyid == 0:
            break
        parent_name = _parent_name
    return parent_name, child_name


def get_gripper_ctrl_indices(phy):
    return [phy.m.actuator(a).id for a in phy.o.rd.gripper_actuator_names]
