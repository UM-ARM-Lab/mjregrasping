from typing import Optional
import rerun as rr

import mujoco
import numpy as np

from mjregrasping.eq_errors import compute_total_eq_error
from mjregrasping.movie import MjMovieMaker
from mjregrasping.params import hp
from mjregrasping.physics import Physics, get_full_q, get_qpos_for_actuators
from mjregrasping.real_val import RealValCommander
from dm_control.mujoco.wrapper.mjbindings import mjlib

import collections

USEFUL_INDICES_vel = [0, 1, 11, 12, 13, 14, 15, 16, 17]
USEFUL_INDICES_pos = [0, 1, 11, 12, 13, 14, 15, 16, 17]
USEFUL_INDICES_ctrl = [0, 1, 9, 10, 11, 12, 13, 14, 15]
def velocity_control(gripper_delta, physics, n_sub_time):
    if len(gripper_delta) < 6:
        gripper_delta = np.concatenate((np.zeros(6 - len(gripper_delta)), gripper_delta), axis=0)
    jac_pos_l = np.zeros((3, physics.model.nv))
    jac_rot_l = np.zeros((3, physics.model.nv))
    jac_pos_r = np.zeros((3, physics.model.nv))
    jac_rot_r = np.zeros((3, physics.model.nv))

    mjlib.mj_jacGeom(physics.model.ptr, physics.data.ptr, jac_pos_l, jac_rot_l, physics.model.name2id('val/left_finger_pad', 'geom'))
    mjlib.mj_jacGeom(physics.model.ptr, physics.data.ptr, jac_pos_r, jac_rot_r, physics.model.name2id('val/right_finger_pad', 'geom'))

    J = np.concatenate((jac_pos_l[:, USEFUL_INDICES_vel], jac_pos_r[:, USEFUL_INDICES_vel]), axis=0)
    J_T = J.T
    ctrl = J_T @ np.linalg.solve(J @ J_T + 1e-6 * np.eye(6), gripper_delta)
    current_qpos = physics.data.qpos[USEFUL_INDICES_pos]

    vmin = physics.model.actuator_ctrlrange[:, 0]
    vmax = physics.model.actuator_ctrlrange[:, 1]
    #Create a list of length 10 that interpolates between the current position and the desired position
    frac = np.linspace(1/n_sub_time, 1, n_sub_time)
    qpos_list = [np.clip(current_qpos + frac[i] * ctrl, vmin, vmax) for i in range(len(frac))]

    return qpos_list

DEFAULT_SUB_TIME_S = 0.1

_INVALID_JOINT_NAMES_TYPE = (
    '`joint_names` must be either None, a list, a tuple, or a numpy array; '
    'got {}.')
_REQUIRE_TARGET_POS_OR_QUAT = (
    'At least one of `target_pos` or `target_quat` must be specified.')

IKResult = collections.namedtuple(
    'IKResult', ['qpos', 'err_norm', 'steps', 'success'])

# ZEROS = np.array([ -.405,   -.214,  -1.571  ,    -1.102,    .831,    .054,    .018,    .64,    .011  ])
# ZEROS = np.array([ -.392,   -.298,  .2  ,    -1.571,    1.305,    -.119,    .119,    .873,    -.024  ])
ZEROS = np.array([ -.155,   -.644,  .12  ,    -1.201,    1.345,    -.188,    .298,    .692,    -.401  ])
# ZEROS = np.array([ 0,   0,  0  ,    -1.571,    0,    0,    0,    0,    0  ])

def qpos_from_site_pose(physics,
                        site_name,
                        target_pos=None,
                        target_quat=None,
                        joint_names=None,
                        tol=1e-3,
                        rot_weight=1.0,
                        regularization_threshold=0.1,
                        regularization_strength=3e-2,
                        jnt_lim_avoidance=.25,
                        max_update_norm=2.0,
                        progress_thresh=20.0,
                        max_steps=100,
                        inplace=False):
  """Find joint positions that satisfy a target site position and/or rotation.

  Args:
    physics: A `mujoco.Physics` instance.
    site_name: A string specifying the name of the target site.
    target_pos: A (3,) numpy array specifying the desired Cartesian position of
      the site, or None if the position should be unconstrained (default).
      One or both of `target_pos` or `target_quat` must be specified.
    target_quat: A (4,) numpy array specifying the desired orientation of the
      site as a quaternion, or None if the orientation should be unconstrained
      (default). One or both of `target_pos` or `target_quat` must be specified.
    joint_names: (optional) A list, tuple or numpy array specifying the names of
      one or more joints that can be manipulated in order to achieve the target
      site pose. If None (default), all joints may be manipulated.
    tol: (optional) Precision goal for `qpos` (the maximum value of `err_norm`
      in the stopping criterion).
    rot_weight: (optional) Determines the weight given to rotational error
      relative to translational error.
    regularization_threshold: (optional) L2 regularization will be used when
      inverting the Jacobian whilst `err_norm` is greater than this value.
    regularization_strength: (optional) Coefficient of the quadratic penalty
      on joint movements.
    max_update_norm: (optional) The maximum L2 norm of the update applied to
      the joint positions on each iteration. The update vector will be scaled
      such that its magnitude never exceeds this value.
    progress_thresh: (optional) If `err_norm` divided by the magnitude of the
      joint position update is greater than this value then the optimization
      will terminate prematurely. This is a useful heuristic to avoid getting
      stuck in local minima.
    max_steps: (optional) The maximum number of iterations to perform.
    inplace: (optional) If True, `physics.data` will be modified in place.
      Default value is False, i.e. a copy of `physics.data` will be made.

  Returns:
    An `IKResult` namedtuple with the following fields:
      qpos: An (nq,) numpy array of joint positions.
      err_norm: A float, the weighted sum of L2 norms for the residual
        translational and rotational errors.
      steps: An int, the number of iterations that were performed.
      success: Boolean, True if we converged on a solution within `max_steps`,
        False otherwise.

  Raises:
    ValueError: If both `target_pos` and `target_quat` are None, or if
      `joint_names` has an invalid type.
  """

  dtype = physics.data.qpos.dtype

  if target_pos is not None and target_quat is not None:
    jac = np.empty((6, physics.model.nv), dtype=dtype)
    err = np.empty(6, dtype=dtype)
    jac_pos, jac_rot = jac[:3], jac[3:]
    err_pos, err_rot = err[:3], err[3:]
  else:
    jac = np.empty((3, physics.model.nv), dtype=dtype)
    err = np.empty(3, dtype=dtype)
    if target_pos is not None:
      jac_pos, jac_rot = jac, None
      err_pos, err_rot = err, None
    elif target_quat is not None:
      jac_pos, jac_rot = None, jac
      err_pos, err_rot = None, err
    else:
      raise ValueError(_REQUIRE_TARGET_POS_OR_QUAT)

  update_nv = np.zeros(physics.model.nv, dtype=dtype)

  if target_quat is not None:
    site_xquat = np.empty(4, dtype=dtype)
    neg_site_xquat = np.empty(4, dtype=dtype)
    err_rot_quat = np.empty(4, dtype=dtype)

  if not inplace:
    physics = physics.copy(share_model=True)

  # Ensure that the Cartesian position of the site is up to date.
  mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

  # Convert site name to index.
  site_id = physics.model.name2id(site_name, 'site')

  # These are views onto the underlying MuJoCo buffers. mj_fwdPosition will
  # update them in place, so we can avoid indexing overhead in the main loop.
  site_xpos = physics.named.data.site_xpos[site_name]
  site_xmat = physics.named.data.site_xmat[site_name]

  # This is an index into the rows of `update` and the columns of `jac`
  # that selects DOFs associated with joints that we are allowed to manipulate.
  if joint_names is None:
    dof_indices = slice(None)  # Update all DOFs.
  elif isinstance(joint_names, (list, np.ndarray, tuple)):
    if isinstance(joint_names, tuple):
      joint_names = list(joint_names)
    # Find the indices of the DOFs belonging to each named joint. Note that
    # these are not necessarily the same as the joint IDs, since a single joint
    # may have >1 DOF (e.g. ball joints).
    indexer = physics.named.model.dof_jntid.axes.row
    # `dof_jntid` is an `(nv,)` array indexed by joint name. We use its row
    # indexer to map each joint name to the indices of its corresponding DOFs.
    dof_indices = indexer.convert_key_item(joint_names)
  else:
    raise ValueError(_INVALID_JOINT_NAMES_TYPE.format(type(joint_names)))

  steps = 0
  success = False

  for steps in range(max_steps):
    err_norm = 0.0

    if target_pos is not None:
      # Translational error.
      err_pos[:] = target_pos - site_xpos
      err_norm += np.linalg.norm(err_pos)
    if target_quat is not None:
      # Rotational error.
      mjlib.mju_mat2Quat(site_xquat, site_xmat)
      mjlib.mju_negQuat(neg_site_xquat, site_xquat)
      mjlib.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
      mjlib.mju_quat2Vel(err_rot, err_rot_quat, 1)
      err_norm += np.linalg.norm(err_rot) * rot_weight
    # print(f'step {steps}, err: {err_norm}')
    
    if err_norm < tol:
    #   logging.debug('Converged after %i steps: err_norm=%3g', steps, err_norm)
      success = True
      break
    else:
      # TODO(b/112141670): Generalize this to other entities besides sites.
      mjlib.mj_jacSite(
          physics.model.ptr, physics.data.ptr, jac_pos, jac_rot, site_id)
      jac_joints = jac[:, dof_indices]

      
      # TODO(b/112141592): This does not take joint limits into consideration.
      reg_strength = (
          regularization_strength if err_norm > regularization_threshold
          else 0.0)
      
      cur_q = physics.data.qpos[dof_indices]
  
      zero_vels = (ZEROS-cur_q) * jnt_lim_avoidance

      J_t = jac_joints.T
      J_pinv = J_t @ np.linalg.inv(jac_joints @ J_t + reg_strength**2 * np.eye(jac_joints.shape[0]))
      update_joints = J_pinv @ err + (np.eye(len(dof_indices)) - J_pinv @ jac_joints) @ zero_vels

      update_norm = np.linalg.norm(update_joints)

      # Check whether we are still making enough progress, and halt if not.
      progress_criterion = err_norm / update_norm
      if progress_criterion > progress_thresh:
        break

      if update_norm > max_update_norm:
        update_joints *= max_update_norm / update_norm

      # Write the entries for the specified joints into the full `update_nv`
      # vector.
      update_nv[dof_indices] = update_joints

      # Update `physics.qpos`, taking quaternions into account.
      mjlib.mj_integratePos(physics.model.ptr, physics.data.qpos, update_nv, 1)

      # Compute the new Cartesian position of the site.
      mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)


  if not inplace:
    # Our temporary copy of physics.data is about to go out of scope, and when
    # it does the underlying mjData pointer will be freed and physics.data.qpos
    # will be a view onto a block of deallocated memory. We therefore need to
    # make a copy of physics.data.qpos while physics.data is still alive.
    qpos = physics.data.qpos.copy()
  else:
    # If we're modifying physics.data in place then it's fine to return a view.
    qpos = physics.data.qpos

  return IKResult(qpos=qpos, err_norm=err_norm, steps=steps, success=success)

def no_results(*args, **kwargs):
    return (None,)

def control_step(phy: Physics, eef_delta_target, sub_time_s: float, mov: Optional[MjMovieMaker] = None,
                 val_cmd: Optional[RealValCommander] = None):
    m = phy.m
    d = phy.d

    n_sub_time = int(sub_time_s / m.opt.timestep)

    if eef_delta_target is not None:
        pass
        # setpoints = velocity_control(eef_delta_target, phy.p, n_sub_time)
    else:
        print("control is None!!!")

    # slow_when_eqs_bad(phy)

    # limit_actuator_windup(phy)

    if mov:
        # This renders every frame
        for _ in range(n_sub_time):
            mujoco.mj_step(m, d, nstep=1)
            mov.render(d)
    else:
        # for i in range(n_sub_time):
        #     phy.p.set_control(setpoints[i])
        #     phy.p.data.qpos[USEFUL_INDICES_pos] = setpoints[i]
        #     phy.p.named.data.qpos['val/rightgripper'] = .5
        #     phy.p.named.data.qpos['val/rightgripper2'] = .5
        #     phy.p.step()
        
        cur_eef_pos = phy.p.named.data.site_xpos['val/right_tool']
        cur_useful_qpos = phy.p.data.qpos[USEFUL_INDICES_pos].copy()

        ik_result = qpos_from_site_pose(phy.p, 'val/right_tool', target_pos=cur_eef_pos + eef_delta_target, 
                                joint_names=['val/joint56', 'val/joint57', 'val/joint1', 'val/joint2', 'val/joint3', 'val/joint4', 'val/joint5', 'val/joint6', 'val/joint7'], 
                                regularization_strength=0, 
                                regularization_threshold=0,
                                jnt_lim_avoidance=.05,
                                max_update_norm=2,
                                max_steps=1000,                         
                                inplace=False)
        if not ik_result.success:
            print('IK failed')
        else:
            # phy.d.ctrl[USEFUL_INDICES_ctrl] = ik_result.qpos[USEFUL_INDICES_pos]

            vmin = phy.p.model.actuator_ctrlrange[USEFUL_INDICES_ctrl, 0]
            vmax = phy.p.model.actuator_ctrlrange[USEFUL_INDICES_ctrl, 1]

            frac = np.linspace(1/(n_sub_time), 1, (n_sub_time))
            qpos_list = [np.clip(cur_useful_qpos * (1-frac[i]) + frac[i] * ik_result.qpos[USEFUL_INDICES_pos], vmin, vmax) for i in range(len(frac))]
            phy.d.ctrl[USEFUL_INDICES_ctrl] = qpos_list[-1]
            # phy.d.ctrl[USEFUL_INDICES_ctrl] = ik_result.qpos[USEFUL_INDICES_pos]
            # phy.p.data.qpos[USEFUL_INDICES_pos] = ik_result.qpos[USEFUL_INDICES_pos]
            for i in range(n_sub_time):
                # phy.p.set_control(qpos_list[i])
                phy.p.data.qpos[USEFUL_INDICES_pos] = qpos_list[i]
                phy.p.named.data.qpos['val/rightgripper'] = .5
                phy.p.named.data.qpos['val/rightgripper2'] = .5
                phy.p.step()
                #Check error of qpos
                # for i in range(100):

                # err = qpos_list[i] - phy.p.data.qpos[USEFUL_INDICES_pos]
                # print('err 0:', np.linalg.norm(err))
                # count = 0
                # while np.linalg.norm(err) > 1e-3 and count < 100:
                #     count += 1
                #     print(f'err {count}:', np.linalg.norm(err))
                #     phy.p.step()
                #     err = qpos_list[i] - phy.p.data.qpos[USEFUL_INDICES_pos]
            for i in range(n_sub_time):
                # phy.p.set_control(qpos_list[i])
                phy.p.data.qpos[USEFUL_INDICES_pos] = qpos_list[-1]
                phy.p.named.data.qpos['val/rightgripper'] = .5
                phy.p.named.data.qpos['val/rightgripper2'] = .5
                phy.p.step()
    if val_cmd:
        mj_q = get_full_q(phy)
        val_cmd.send_pos_command(mj_q, slow=slow)
        # val_cmd.pull_rope_towards_cdcpd(phy, n_sub_time / 4)


def slow_when_eqs_bad(phy):
    speed_factor = get_speed_factor(phy)
    phy.d.ctrl *= speed_factor


def get_speed_factor(phy):
    return 1
    total_eq_error = compute_total_eq_error(phy)
    speed_factor = np.clip(np.exp(-700 * total_eq_error), 0, 1)
    return speed_factor


def limit_actuator_windup(phy):
    qpos_for_act = get_qpos_for_actuators(phy)
    phy.d.act = qpos_for_act + np.clip(phy.d.act - qpos_for_act, -hp['act_windup_limit'], hp['act_windup_limit'])
