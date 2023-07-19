import mujoco
import transformations
from scipy.linalg import logm, block_diag
import numpy as np
import rerun as rr

from arc_utilities import ros_init
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.homotopy_utils import make_ring_skeleton, skeleton_field_dir
from mjregrasping.ik import full_jacobian
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.my_transforms import angle_between
from mjregrasping.params import Params
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rollout import control_step, DEFAULT_SUB_TIME_S
from mjregrasping.rviz import MjRViz
from mjregrasping.scenarios import val_untangle
from mjregrasping.viz import make_viz, Viz


@ros_init.with_ros("jac")
def main():
    np.set_printoptions(precision=5, suppress=True, linewidth=200)

    rr.init('low_level_grasping')
    rr.connect()

    xml_path = 'panda_nohand.xml'
    tfw = TF2Wrapper()
    mjviz = MjRViz(xml_path, tfw)
    p = Params()
    mjrr=MjReRun(xml_path)
    viz = Viz(rviz=mjviz, mjrr=mjrr, tfw=tfw, p=p)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    d.qpos[3] = -np.pi/4
    phy = Physics(m, d)

    mujoco.mj_forward(m, d)

    Jp = np.zeros((3, phy.m.nv))
    Jr = np.zeros((3, phy.m.nv))
    mujoco.mj_jacBody(phy.m, phy.d, Jp, Jr, m.body("attachment").id)

    mjrr.viz(phy)

    print(Jp.T)


if __name__ == '__main__':
    main()
