import logging

import mujoco

logger = logging.getLogger(f'rosout.{__name__}')


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
