#!/usr/bin/env python3
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import rerun as rr

from arc_utilities import ros_init
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import activate_grasp
from mjregrasping.move_to_joint_config import pid_to_joint_config, execute_grasp_plan
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.physics import Physics, get_q
from mjregrasping.rollout import DEFAULT_SUB_TIME_S
from mjregrasping.rrt import GraspRRT
from mjregrasping.rviz import MujocoXmlExpander
from mjregrasping.scenarios import val_untangle, get_untangle_skeletons
from mjregrasping.teleport_to_plan import teleport_to_end_of_plan
from mjregrasping.trials import save_trial
from mjregrasping.viz import make_viz, Viz
from moveit_msgs.msg import MoveItErrorCodes


def randomize_rack(orignial_path: Path, rng: np.random.RandomState):
    # load xml
    parser = MujocoXmlExpander(orignial_path)
    root = parser.xml_root

    # iterate over all worldbody elements
    # and all body elements within them recursively
    def _get_e(tag: str, name: str):
        for e in root.iter(tag):
            if "name" in e.attrib and e.attrib["name"] == name:
                return e

    def _get_vec(e, k: str):
        x_str = e.attrib[k]
        x = [float(x) for x in x_str.split()]
        return x

    def _set_vec(e, x, k: str):
        x_str = " ".join([str(x) for x in x])
        e.attrib[k] = x_str

    def _set_vec_i(e, k: str, i: int, x_i: float):
        x = _get_vec(e, k)
        x[i] = x_i
        _set_vec(e, x, k)

    attach = _get_e("body", "attach")
    attach_pos = _get_vec(attach, 'pos')
    attach_pos[0] += rng.uniform(-0.05, 0.05)
    attach_pos[1] += rng.uniform(-0.05, 0.05)
    _set_vec(attach, attach_pos, 'pos')

    # shrink the computer rack in the X axis
    # h will be the half-size in X of the bottom/top geoms of the rack
    h = rng.uniform(0.2, 0.45)
    rack1_bottom = _get_e("geom", "rack1_bottom")
    rack1_post1 = _get_e("geom", "rack1_post1")
    post_size = _get_vec(rack1_post1, 'size')
    rack1_post2 = _get_e("geom", "rack1_post2")
    rack1_post3 = _get_e("geom", "rack1_post3")
    rack1_post4 = _get_e("geom", "rack1_post4")
    rack2_bottom = _get_e("geom", "rack2_bottom")
    rack2_top = _get_e("geom", "rack2_top")
    rack2_post1 = _get_e("geom", "rack2_post1")
    rack2_post2 = _get_e("geom", "rack2_post2")
    rack2_post3 = _get_e("geom", "rack2_post3")
    rack2_post4 = _get_e("geom", "rack2_post4")
    _set_vec_i(rack1_bottom, 'size', 1, h + post_size[1] - 0.001)
    _set_vec_i(rack2_bottom, 'size', 1, h)
    _set_vec_i(rack2_top, 'size', 1, h + post_size[1] - 0.001)
    _set_vec_i(rack1_post1, 'pos', 1, -h)
    _set_vec_i(rack1_post2, 'pos', 1, h)
    _set_vec_i(rack1_post3, 'pos', 1, -h)
    _set_vec_i(rack1_post4, 'pos', 1, h)
    _set_vec_i(rack2_post1, 'pos', 1, -h)
    _set_vec_i(rack2_post2, 'pos', 1, h)
    _set_vec_i(rack2_post3, 'pos', 1, -h)
    _set_vec_i(rack2_post4, 'pos', 1, h)

    # compute the extents of the rack for placing the box and computer on it
    rack1_bottom_pos = _get_vec(rack1_bottom, 'pos')
    bottom_size = _get_vec(rack1_bottom, 'size')

    x0 = rack1_bottom_pos[0] - bottom_size[0] / 2
    x1 = rack1_bottom_pos[0] + bottom_size[0] / 2
    y0 = rack1_bottom_pos[1] - bottom_size[1] / 2
    y1 = rack1_bottom_pos[1] + bottom_size[1] / 2

    # place the box on the rack
    box1 = _get_e("geom", "box1")
    box1_size = _get_vec(box1, 'size')
    box1_x0 = x0 + box1_size[0] / 2
    box1_x1 = x1 - box1_size[0] / 2
    box1_y0 = y0 + box1_size[1] / 2
    box1_y1 = y1 - box1_size[1] / 2
    _set_vec_i(box1, 'pos', 0, rng.uniform(box1_x0, box1_x1))
    _set_vec_i(box1, 'pos', 1, rng.uniform(box1_y0, box1_y1))

    # place the computer on the rack
    case = _get_e("geom", "case")
    case_size = _get_vec(case, 'size')
    computer = _get_e("body", "computer")
    computer_pos = _get_vec(computer, 'pos')
    c_y0 = -h + case_size[1] / 2
    c_y1 = h - case_size[1] / 2
    computer_pos[1] = rng.uniform(c_y0, c_y1)
    _set_vec(computer, computer_pos, 'pos')

    # save to new temp xml file and return the path
    new_filename = 'models/tmp.xml'
    parser.xml_tree.write(new_filename)
    return Path(new_filename)


def randomize_qpos(phy: Physics, rng: np.random.RandomState, viz: Optional[Viz]):
    q = get_q(phy)
    q[0] += np.deg2rad(rng.uniform(-5, 5))
    q[1] += np.deg2rad(rng.uniform(-5, 5))
    pid_to_joint_config(phy, viz, q, sub_time_s=DEFAULT_SUB_TIME_S)


@ros_init.with_ros("randomize_untangle")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('randomize_untangle')
    rr.connect()

    scenario = val_untangle

    viz = make_viz(scenario)

    root = Path("trial_data") / scenario.name
    root.mkdir(exist_ok=True, parents=True)

    grasp_rrt = GraspRRT()

    rng = np.random.RandomState(0)
    for trial_idx in range(10):
        good = False
        while not good:
            # Configure the model before we construct the data and physics object
            new_path = randomize_rack(scenario.xml_path, rng)

            m = mujoco.MjModel.from_xml_path(str(new_path))

            d = mujoco.MjData(m)
            objects = MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
            phy = Physics(m, d, objects)
            # Must call mj_forward or the skeletons will be wrong
            mujoco.mj_forward(m, d)

            skeletons = get_untangle_skeletons(phy)
            viz.skeletons(skeletons)

            rope_xyz_q_indices = phy.o.rope.qpos_indices[:3]
            phy.d.qpos[rope_xyz_q_indices] = phy.d.body("attach").xpos
            phy.m.eq("attach").data[3:6] = 0
            phy.d.qpos[phy.m.joint("joint56").qposadr[0]] = np.deg2rad(-30)
            phy.d.act[phy.m.actuator("joint56_vel").id] = np.deg2rad(-30)
            phy.d.qpos[phy.m.joint("joint57").qposadr[0]] = np.deg2rad(35)
            phy.d.act[phy.m.actuator("joint57_vel").id] = np.deg2rad(35)
            phy.d.qpos[phy.m.joint("joint1").qposadr[0]] = np.deg2rad(-30)
            phy.d.act[phy.m.actuator("joint1_vel").id] = np.deg2rad(-30)
            phy.d.qpos[phy.m.joint("joint41").qposadr[0]] = np.deg2rad(-30)
            phy.d.act[phy.m.actuator("joint41_vel").id] = np.deg2rad(-30)

            mujoco.mj_step(phy.m, phy.d, 500)
            viz.viz(phy, is_planning=True)

            for j in range(5):
                loc = rng.uniform(0.5, 1.0)
                grasp_rrt.fix_start_state_in_place(phy, viz)
                res, scene_msg = grasp_rrt.plan(phy, [Strategies.STAY, Strategies.NEW_GRASP], [-1, loc], viz, pos_noise=0.2)
                if res.error_code.val == MoveItErrorCodes.SUCCESS:
                    teleport_to_end_of_plan(phy, res)
                    activate_grasp(phy, 'right', loc)

                    randomize_qpos(phy, rng, viz)

                    q = input("is this good? (y/n) ")
                    if q == "y":
                        good = True
                        break

        save_trial(trial_idx, phy, scenario, None, skeletons)
        print(f"Trial {trial_idx} saved")


if __name__ == "__main__":
    main()
