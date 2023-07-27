import itertools
import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import Event, Process, Queue
from time import perf_counter, sleep
from typing import Callable, Optional

import mujoco
import numpy as np
import pybio_ik
import rerun as rr
import transformations
from networkx import is_connected
from sklearn.neighbors import KDTree

from arc_utilities import ros_init
from mjregrasping.geometry import pairwise_squared_distances
from mjregrasping.goal_funcs import get_contact_cost
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.physics import Physics
from mjregrasping.rviz import make_clear_marker
from mjregrasping.scenarios import val_untangle
from mjregrasping.viz import Viz, make_viz


def rrt(phy: Physics,
        q0: np.ndarray,
        goal,
        rng: np.random.RandomState,
        state_checker: Callable,
        run_goal_sampler: Callable,
        goal_checker: Callable,
        viz: Optional[Viz],
        dq_step_size=0.1,
        extension_max_dx=1,
        goal_bias=0.10,
        min_nodes=100,
        max_time_s=30.0):
    qs = [q0]
    parent_ids = [None]

    kd = KDTree(qs)
    kd_time = 0
    state_checking_time = 0
    fk_time = 0

    goal_queue = Queue()
    done = Event()
    goal_sampler_process = Process(target=run_goal_sampler, args=(done, goal_queue, phy, goal))
    goal_sampler_process.start()
    goal_samples = []

    def _stop_and_cleanup():
        done.set()
        while not goal_queue.empty():
            goal_queue.get()
        goal_samples.clear()
        goal_sampler_process.join()

    i = 0
    planning_t0 = perf_counter()
    while (perf_counter() - planning_t0) < max_time_s or len(qs) < min_nodes:
        if not goal_queue.empty():
            goal_q = goal_queue.get(block=False)
            goal_samples.append(goal_q)

        sample_goal = len(goal_samples) > 0 and rng.uniform(0, 1) < goal_bias
        if sample_goal:
            q_rand = goal_samples[rng.randint(0, len(goal_samples))]
        else:
            q_rand = uniform_rand_q(phy.m, rng)

        if viz.p.viz_goal_samples and sample_goal:
            set_q_and_do_fk(phy, q_rand)
            viz.viz(phy, is_planning=True)

        t0 = perf_counter()
        _, parent_id = kd.query(q_rand[None], k=1)
        kd_time += perf_counter() - t0
        parent_id = int(parent_id[0, 0])

        q_near = qs[parent_id]

        if viz.p.viz_q_near:
            set_q_and_do_fk(phy, q_near)
            viz.viz(phy, is_planning=True)

        d_total = min(np.linalg.norm(q_rand - q_near), extension_max_dx)
        n_step = max(int(d_total / dq_step_size), 1)

        q_new = None
        for d in np.linspace(dq_step_size, d_total, n_step):
            q_tmp = q_near + d * (q_rand - q_near) / d_total

            t0 = perf_counter()
            set_q_and_do_fk(phy, q_tmp)
            fk_time += perf_counter() - t0

            t0 = perf_counter()
            state_valid = state_checker(phy, q_tmp, viz)
            state_checking_time += perf_counter() - t0
            if not state_valid:
                break
            q_new = q_tmp

        if q_new is not None:
            qs.append(q_new)
            parent_ids.append(parent_id)

            if goal_checker(phy, goal):
                print("Found a path!")
                path = [q_new]
                while parent_id is not None:
                    path.append(qs[parent_id])
                    parent_id = parent_ids[parent_id]
                path = np.array(path[::-1])

                print(f'{kd_time=:.3f} {fk_time=:.3f} {state_checking_time=:.3f} {len(qs)=}')

                _stop_and_cleanup()
                return path

            t0 = perf_counter()
            kd = KDTree(qs)
            kd_time += perf_counter() - t0
        i += 1

    print(f'{kd_time=:.3f}, {fk_time=:.3f}, {state_checking_time=:.3f} {len(qs)=}')

    _stop_and_cleanup()
    return None


def run_bio_ik_goal_sampler(done: Event,
                            queue: Queue,
                            phy: Physics,
                            goal_tool_positions,
                            mujoco2moveit):
    # These two objects must be initialized in the new process, you can't copy them over from the parent even though
    # they are picklable, the ros subscribers will break silently.
    ik = pybio_ik.BioIK("hdt_michigan/robot_description")

    goal_tool_positions_h = np.concatenate((goal_tool_positions, np.ones([2, 1])), axis=-1).T
    goal_tool_positions_moveit = (mujoco2moveit @ goal_tool_positions_h)[:3].T

    grippers_qs = []
    gripper_q_ids = []
    for act_name in phy.o.rd.gripper_actuator_names:
        act = phy.m.actuator(act_name)
        grippers_qs.append(np.arange(*act.actrange, step=np.deg2rad(10)).tolist())
        gripper_q_ids.append(act.trnid[0])

    while not done.is_set():
        phy_i = phy.copy_data()

        rng = np.random.RandomState(0)
        ik_seed_q = duplicate_gripper_qs(uniform_rand_q(phy_i.m, rng))

        for _ in range(10):
            # use bio IK to find a joint configuration that satisfies the goal tool positions
            q_duplicated = ik.ik_from(targets=dict(zip(phy_i.o.rd.tool_sites, goal_tool_positions_moveit)),
                                      start=ik_seed_q,
                                      group_name='whole_body')

            if not q_duplicated:
                continue

            q = deduplicate_gripper_qs(q_duplicated)

            # discretize gripper q's because they don't change the tool position,
            # but they may matter for collision checking
            for gripper_qs in itertools.product(*grippers_qs):
                # FIXME: hard-coded for val
                q[9] = gripper_qs[0]
                q[17] = gripper_qs[1]

                set_q_and_do_fk(phy_i, q)

                contact_cost = get_contact_cost(phy_i)

                if contact_cost > 0:
                    # Clever trick -- the simulator will resolve contacts in a way that doesn't push us too far away
                    # so if we step the simulation, the resulting q is probably a good seed for our next IK attempt
                    mujoco.mj_step(phy_i.m, phy_i.d, nstep=2)
                    ik_seed_q = phy_i.d.qpos[phy_i.o.robot.qpos_indices].copy()
                    ik_seed_q += rng.normal(0, 0.01, size=ik_seed_q.shape)
                    continue

                queue.put(q)
    queue.close()


def contact_state_checker(phy: Physics, q, viz: Optional[Viz]):
    if viz.p.viz_state_checker:
        viz.viz(phy, is_planning=True)
    contact_cost = get_contact_cost(phy)

    return contact_cost == 0


def uniform_rand_q(m, rng):
    q_low = m.actuator_actrange[:, 0]
    q_high = m.actuator_actrange[:, 1]
    q_rand = rng.uniform(q_low, q_high)
    return q_rand


def position_goal_checker(phy, goal_tool_positions, pos_tol):
    contact_cost = get_contact_cost(phy)
    if contact_cost > 0:
        return False

    goal_satisfied = True
    for tool_site_name, goal_pos in zip(phy.o.rd.tool_sites, goal_tool_positions):
        tool_site_xpos = phy.d.site(tool_site_name).xpos
        d = np.linalg.norm(goal_pos - tool_site_xpos)
        if d > pos_tol:
            goal_satisfied = False
            break

    return goal_satisfied


def set_q_and_do_fk(phy, q):
    # do FK and check for contacts
    q_duplicated = duplicate_gripper_qs(q)
    phy.d.qpos[phy.o.robot.qpos_indices] = q_duplicated #.copy()
    mujoco.mj_forward(phy.m, phy.d)


def duplicate_gripper_qs(q):
    q_duplicated = np.insert(q, 9, q[9])
    q_duplicated = np.insert(q_duplicated, 18, q[17])
    return q_duplicated


def deduplicate_gripper_qs(q_duplicated):
    q = np.delete(q_duplicated, 18)
    q = np.delete(q, 9)
    return q


@ros_init.with_ros("test_rrt")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rr.init('test_rrt')
    rr.connect()

    scenario = val_untangle
    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))

    viz = make_viz(scenario)
    phy = Physics(m, mujoco.MjData(m),
                  objects=MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))
    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)

    rng = np.random.RandomState(0)

    # these are in mujoco world frame
    goal_tool_positions = np.array([
        [0.75, -0.20, 0.75],
        [0.75, -0.20, 0.33],
    ])

    mujoco2moveit = transformations.compose_matrix(angles=[0, 0, 0], translate=-phy.m.body("val_base").pos)

    clear_all_marker = make_clear_marker()
    for _ in range(5):
        viz.markers_pub.publish(clear_all_marker)
        sleep(0.01)

    path = rrt(phy=phy, q0=np.zeros(18),
               goal=goal_tool_positions,
               rng=rng,
               state_checker=contact_state_checker,
               run_goal_sampler=partial(run_bio_ik_goal_sampler, mujoco2moveit=mujoco2moveit),
               goal_checker=partial(position_goal_checker, pos_tol=0.01),
               max_time_s=15.0,
               viz=viz)
    # path = prm(phy=phy, q0=np.zeros(18),
    #            goal=goal_tool_positions,
    #            rng=rng,
    #            state_checker=contact_state_checker,
    #            goal_sampler=partial(bio_ik_goal_sampler, mujoco2moveit=mujoco2moveit),
    #            goal_checker=position_goal_checker,
    #            max_time_s=30.0,
    #            viz=viz)

    if path is None:
        print("Failed to find a path")
        return

    # should up-sample the path for visualization purposes

    for point in path:
        set_q_and_do_fk(phy, point)
        viz.viz(phy, is_planning=True, detailed=True)
        sleep(0.1)


if __name__ == '__main__':
    main()
