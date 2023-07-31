import itertools
from functools import partial
from multiprocessing import Event, Process, Queue
from time import perf_counter, sleep
from typing import Callable, Optional, Dict

import mujoco
import numpy as np
import pybio_ik
import rerun as rr
from sklearn.neighbors import KDTree

from arc_utilities import ros_init
from mjregrasping.goal_funcs import get_contact_cost
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.physics import Physics
from mjregrasping.rviz import make_clear_marker
from mjregrasping.scenarios import val_untangle
from mjregrasping.viz import Viz, make_viz

mujoco2moveit = np.array([[1., 0., 0., 0.],
                          [0., 1., 0., 0.],
                          [0., 0., 1., -0.145],
                          [0., 0., 0., 1.]])


def to_moveit(p):
    return (mujoco2moveit @ np.concatenate([p, np.ones(1)]))[:3]


def rrt(phy: Physics,
        q0: np.ndarray,
        joint_ids: np.ndarray,
        group_name,
        goal,
        rng: np.random.RandomState,
        state_checker: Callable,
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
    goal_sampler_process = Process(target=run_bio_ik_goal_sampler,
                                   args=(done, goal_queue, phy, joint_ids, group_name, goal))
    # goal_sampler_process.start()
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
            q_rand = uniform_rand_q(phy, joint_ids, rng)

        if viz.p.viz_goal_samples and sample_goal:
            set_q_and_do_fk(phy, joint_ids, q_rand)
            viz.viz(phy, is_planning=True)

        t0 = perf_counter()
        _, parent_id = kd.query(q_rand[None], k=1)
        kd_time += perf_counter() - t0
        parent_id = int(parent_id[0, 0])

        q_near = qs[parent_id]

        if viz.p.viz_q_near:
            set_q_and_do_fk(phy, joint_ids, q_near)
            viz.viz(phy, is_planning=True)

        d_total = min(np.linalg.norm(q_rand - q_near), extension_max_dx)
        n_step = max(int(d_total / dq_step_size), 1)

        q_new = None
        for d in np.linspace(dq_step_size, d_total, n_step):
            q_tmp = q_near + d * (q_rand - q_near) / d_total

            t0 = perf_counter()
            set_q_and_do_fk(phy, joint_ids, q_tmp)
            fk_time += perf_counter() - t0

            t0 = perf_counter()
            state_valid = state_checker(phy, viz)
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
                            joint_ids,
                            group_name,
                            goal_tool_positions: Dict[str, np.ndarray],
                            ):
    # These two objects must be initialized in the new process, you can't copy them over from the parent even though
    # they are picklable, the ros subscribers will break silently.
    ik = pybio_ik.BioIK("hdt_michigan/robot_description")

    goal_tool_positions_moveit = {k: to_moveit(p) for k, p in goal_tool_positions.items()}

    gripper_q_ids = get_gripper_qpos_indices(phy)
    grippers_qs = [[0, 0.2, 0.4]] * len(gripper_q_ids)

    rng = np.random.RandomState(0)
    while not done.is_set():
        phy_i = phy.copy_data()

        ik_seed_q = duplicate_gripper_qs(phy, joint_ids, uniform_rand_q(phy_i, joint_ids, rng))

        for _ in range(10):
            # use bio IK to find a joint configuration that satisfies the goal tool positions
            q_duplicated = ik.ik_from(targets=goal_tool_positions_moveit, start=ik_seed_q, group_name=group_name)

            if not q_duplicated:
                continue

            q = deduplicate_gripper_qs(phy.m, joint_ids, q_duplicated)

            # discretize gripper q's because they don't change the tool position,
            # but they may matter for collision checking
            for gripper_qs in itertools.product(*grippers_qs):
                # FIXME: hard-coded for val
                q[9] = gripper_qs[0]
                q[17] = gripper_qs[1]

                set_q_and_do_fk(phy_i, joint_ids, q)

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


def contact_state_checker(phy: Physics, viz: Optional[Viz]):
    if viz.p.viz_state_checker:
        viz.viz(phy, is_planning=True)
    contact_cost = get_contact_cost(phy)

    return contact_cost == 0


def position_goal_checker(phy, goal_tool_positions, pos_tol):
    contact_cost = get_contact_cost(phy)
    if contact_cost > 0:
        return False

    goal_satisfied = True
    for tool_site_name, goal_pos in goal_tool_positions.items():
        tool_site_xpos = phy.d.site(tool_site_name).xpos
        d = np.linalg.norm(goal_pos - tool_site_xpos)
        if d > pos_tol:
            goal_satisfied = False
            break

    return goal_satisfied


def set_q_and_do_fk(phy, joint_ids, q):
    # do FK and check for contacts
    q_duplicated = duplicate_gripper_qs(phy, joint_ids, q)
    phy.d.qpos[phy.o.robot.qpos_indices] = q_duplicated  # .copy()
    mujoco.mj_forward(phy.m, phy.d)


def uniform_rand_q(phy, joint_ids, rng):
    q_low = phy.m.actuator_actrange[:, 0]
    q_high = phy.m.actuator_actrange[:, 1]
    q_rand = rng.uniform(q_low, q_high)
    q_rand = duplicate_gripper_qs(phy, joint_ids, q_rand)
    q_rand = q_rand[joint_ids]
    return q_rand


def duplicate_gripper_qs(phy, joint_ids, q):
    gripper_q_ids = get_gripper_qpos_indices(phy)
    for gripper_q_id in gripper_q_ids:
        if gripper_q_id in joint_ids:
            j = np.where(joint_ids == gripper_q_id)[0][0]
            q = np.insert(q, j + 1, q[j])
    return q


def deduplicate_gripper_qs(m, joint_ids, q_duplicated):
    q = np.delete(q_duplicated, 18)
    q = np.delete(q, 9)
    return q


def get_gripper_qpos_indices(phy):
    gripper_q_ids = []
    for act_name in phy.o.rd.gripper_actuator_names:
        act = phy.m.actuator(act_name)
        gripper_q_ids.append(act.trnid[0])
    return gripper_q_ids


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

    group_name = "whole_body"

    # these are in mujoco world frame
    goal_tool_positions = {
        'left_tool':  np.array([0.75, -0.20, 0.75]),
        'right_tool': np.array([0.75, -0.20, 0.33]),
    }

    clear_all_marker = make_clear_marker()
    for _ in range(5):
        viz.markers_pub.publish(clear_all_marker)
        sleep(0.01)

    # group = MoveGroupCommander(group_name, robot_description="hdt_michigan/robot_description", ns="hdt_michigan")
    # mj_group_joint_ids = np.array([phy.o.robot.joint_names.index(n) for n in group.get_active_joints()])
    # dq_step_size = 0.1
    # # TODO: use the move group
    # path = rrt(phy=phy,
    #            q0=np.zeros(len(mj_group_joint_ids)),
    #            joint_ids=mj_group_joint_ids,
    #            group_name=group_name,
    #            goal=goal_tool_positions,
    #            rng=rng,
    #            state_checker=contact_state_checker,
    #            goal_checker=partial(position_goal_checker, pos_tol=0.02),
    #            dq_step_size=dq_step_size,
    #            max_time_s=30.0,
    #            viz=viz)

    if path is None:
        print("Failed to find a path")
        return

    # should up-sample the path for visualization purposes
    dense_path = []
    for i in range(len(path) - 1):
        q0 = path[i]
        q1 = path[i + 1]
        d = np.linalg.norm(q1 - q0)
        n_step = max(int(d / dq_step_size), 1)
        for j in range(n_step):
            dense_path.append(q0 + j * (q1 - q0) / n_step)
    dense_path.append(path[-1])

    for point in dense_path:
        set_q_and_do_fk(phy, mj_group_joint_ids, point)
        viz.viz(phy, is_planning=True)


if __name__ == '__main__':
    main()
