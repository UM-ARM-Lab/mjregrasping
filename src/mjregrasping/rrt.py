from time import perf_counter, sleep

import mujoco
import transformations
import pybio_ik
from sklearn.neighbors import KDTree
import rerun as rr
import numpy as np
from typing import Callable, Optional

from arc_utilities import ros_init
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.goal_funcs import get_contact_cost
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.params import Params
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rviz import MjRViz
from mjregrasping.scenarios import val_untangle
from mjregrasping.viz import Viz


class Node:

    def __init__(self, q):
        self.q = q
        self.children = []


def plan_from_q_to_target_pos(phy: Physics,
                              q0: np.ndarray,
                              rng: np.random.RandomState,
                              state_checker: Callable,
                              goal_sampler: Callable,
                              goal_checker: Callable,
                              viz: Optional[Viz],
                              dq_step_size=0.5,
                              goal_bias=0.05,
                              max_time_s=10.0):
    qs = [q0]
    parent_ids = [None]
    q_low = phy.m.actuator_actrange[:, 0]
    q_high = phy.m.actuator_actrange[:, 1]

    kd = KDTree(qs)

    i = 0
    t0 = perf_counter()
    while (perf_counter() - t0) < max_time_s:
        if i == 0 or rng.uniform(0, 1) < goal_bias:
            q_rand = goal_sampler(phy, viz)
            if q_rand is None:
                print("Goal sampler failed, using a random q")
                q_rand = rng.uniform(q_low, q_high)
        else:
            q_rand = rng.uniform(q_low, q_high)

        _, parent_id = kd.query(q_rand[None], k=1)
        parent_id = int(parent_id[0, 0])
        q_near = qs[parent_id]

        d_total = np.linalg.norm(q_rand - q_near)
        n_step = max(int(d_total / dq_step_size), 1)

        for d in np.linspace(dq_step_size, d_total, n_step):
            q_new = q_near + d * (q_rand - q_near) / d_total

            if state_checker(phy, q_new, viz):
                qs.append(q_new)
                parent_ids.append(parent_id)

                if goal_checker(q_new, viz):
                    # extract path
                    path = [q_new]
                    while parent_id is not None:
                        path.append(qs[parent_id])
                        parent_id = parent_ids[parent_id]
                    path = np.array(path[::-1])
                    return path
            else:
                break

            parent_id += 1

        kd = KDTree(qs)
        i += 1

    return None


def state_checker(phy: Physics, q, viz: Optional[Viz]):
    set_q_and_do_fk(phy, q)
    if viz:
        viz.viz(phy, is_planning=True)
    contact_cost = get_contact_cost(phy)

    return contact_cost == 0


def set_q_and_do_fk(phy, q):
    # do FK and check for contacts
    q_duplicated = duplicate_gripper_qs(q)
    phy.d.qpos[phy.o.robot.qpos_indices] = q_duplicated
    mujoco.mj_forward(phy.m, phy.d)


def duplicate_gripper_qs(q):
    q_duplicated = np.insert(q, 9, q[9])
    q_duplicated = np.insert(q_duplicated, 18, q[17])
    return q_duplicated


def deduplicate_gripper_qs(q_duplicated):
    q = np.delete(q_duplicated, 18)
    q = np.delete(q, 9)
    return q


@ros_init.with_ros("test_regrasp_homotopy")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rr.init('test_regrasp_homotopy')
    rr.connect()

    scenario = val_untangle
    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))

    tfw = TF2Wrapper()
    mjviz = MjRViz(scenario.xml_path, tfw)
    p = Params()

    viz = Viz(rviz=mjviz, mjrr=MjReRun(scenario.xml_path), tfw=tfw, p=p)
    phy = Physics(m, mujoco.MjData(m),
                  objects=MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))
    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)

    rng = np.random.RandomState(0)

    ik = pybio_ik.BioIK("hdt_michigan/robot_description")

    # these are in mujoco world frame
    goal_tool_positions = np.array([
        [0.68, 0.25, 0.75],
        [0.68, -0.25, 0.75],
    ])

    mujoco2moveit = transformations.compose_matrix(angles=[0, 0, 0], translate=-phy.m.body("val_base").pos)
    goal_tool_positions_h = np.concatenate((goal_tool_positions, np.ones([2, 1])), axis=-1).T
    goal_tool_positions_moveit = (mujoco2moveit @ goal_tool_positions_h)[:3].T

    def goal_sampler(phy, viz: Optional[Viz]):
        for _ in range(100):
            # use bio IK to find a joint configuration that satisfies the goal tool positions
            q_duplicated = ik.ik(dict(zip(phy.o.rd.tool_sites, goal_tool_positions_moveit)), 'whole_body')

            if not q_duplicated:
                continue
            q = deduplicate_gripper_qs(q_duplicated)
            set_q_and_do_fk(phy, q)

            contact_cost = get_contact_cost(phy)
            if contact_cost > 0:
                continue

            if viz:
                viz.viz(phy, is_planning=True)

            return q
        return None

    def goal_checker(q, viz: Optional[Viz]):
        set_q_and_do_fk(phy, q)
        if viz:
            viz.viz(phy, is_planning=True)

        contact_cost = get_contact_cost(phy)
        if contact_cost > 0:
            return False

        goal_satisfied = True
        for tool_site_name, goal_pos in zip(phy.o.rd.tool_sites, goal_tool_positions):
            tool_site_xpos = phy.d.site(tool_site_name).xpos
            d = np.linalg.norm(goal_pos - tool_site_xpos)
            if d > 0.05:
                goal_satisfied = False
                break

        return goal_satisfied

    path = plan_from_q_to_target_pos(phy, np.zeros(18), rng, state_checker, goal_sampler, goal_checker, None)

    for point in path:
        set_q_and_do_fk(phy, point)
        viz.viz(phy, is_planning=True)
        sleep(0.5)


if __name__ == '__main__':
    main()
