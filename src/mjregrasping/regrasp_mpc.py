from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import rerun as rr
from matplotlib import cm

import rospy
from mjregrasping.buffer import Buffer
from mjregrasping.grasping import get_grasp_locs
from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner, release_and_settle, grasp_and_settle, \
    get_geodesic_dist
from mjregrasping.move_to_joint_config import execute_grasp_plan
from mjregrasping.movie import MjMovieMaker
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.regrasp_goal import RegraspGoal, GraspGoal
from mjregrasping.regrasping_mppi import RegraspMPPI, do_grasp_dynamics, rollout
from mjregrasping.robot_data import val
from mjregrasping.rollout import control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.sdf_collision_checker import SDFCollisionChecker
from mjregrasping.viz import Viz


class RegraspMPC:

    def __init__(self, pool: ThreadPoolExecutor, mppi_nu: int, skeletons, sdf, goal, seed: int, viz: Viz,
                 mov: Optional[MjMovieMaker] = None):
        self.mppi_nu = mppi_nu
        self.skeletons = skeletons
        self.sdf = sdf
        self.pool = pool
        self.viz = viz
        self.op_goal = goal
        self.mov = mov
        self.is_gasping_rng = np.random.RandomState(0)

        self.mppi = RegraspMPPI(pool=self.pool, nu=self.mppi_nu, seed=seed, horizon=hp['regrasp_horizon'],
                                noise_sigma=val.noise_sigma, temp=hp['regrasp_temp'])
        self.state_history = Buffer(hp['state_history_size'])
        self.max_dq = None
        self.reset_trap_detection()

    def reset_trap_detection(self):
        self.state_history.reset()
        self.max_dq = 0

    def run(self, phy: Physics):
        num_samples = hp['regrasp_n_samples']
        grasp_rrt = GraspRRT()

        cc = SDFCollisionChecker(self.sdf)
        planner = HomotopyRegraspPlanner(self.op_goal, grasp_rrt, self.skeletons, cc)

        grasp_goal = GraspGoal(get_grasp_locs(phy))

        regrasp_goal = RegraspGoal(self.op_goal, grasp_goal, hp['grasp_goal_radius'], self.viz)
        regrasp_goal.viz_goal(phy)

        self.mppi.reset()
        self.reset_trap_detection()

        itr = 0
        max_iters = 5000
        command = None
        sub_time_s = None
        self.viz.viz(phy)
        while True:
            if rospy.is_shutdown():
                self.mov.close()
                raise RuntimeError("ROS shutdown")

            if itr > max_iters:
                return False

            regrasp_goal.viz_goal(phy)
            if regrasp_goal.satisfied(phy):
                print("Goal reached!")
                return True

            stuck_frac = self.check_is_stuck(phy)
            is_stuck_vec = np.array([stuck_frac, stuck_frac]) < hp['frac_dq_threshold']
            needs_reset = False
            if np.any(is_stuck_vec):
                print("Stuck! Replanning...")
                initial_geodesic_cost = get_geodesic_dist(grasp_goal.get_grasp_locs(), self.op_goal)
                needs_reset = True
                new_locs, _, strategy = planner.generate(phy, self.viz)
                new_geodesic_cost = get_geodesic_dist(new_locs, self.op_goal)

                # if we are unable to improve by grasping closer to the keypoint, update the blacklist and replan
                if new_geodesic_cost >= initial_geodesic_cost:
                    print("Unable to improve by grasping closer to the keypoint. Updating blacklist and replanning...")
                    planner.update_blacklists(phy)
                    new_locs, _, strategy = planner.generate(phy, self.viz)

                # now execute the plan
                release_and_settle(phy, strategy, self.viz, is_planning=False)
                # Since it's a pain to get the previously computed plan out of the bayes-opt planner, just recompute it
                sln = planner.get_sln(new_locs, strategy)
                res = sln.res
                qs = np.array([p.positions for p in res.trajectory.joint_trajectory.points])
                execute_grasp_plan(phy, qs, self.viz, is_planning=False)
                grasp_and_settle(phy, new_locs, self.viz, is_planning=False)

                grasp_goal.set_grasp_locs(new_locs)
                # save_data_and_eq(phy, Path(f'states/on_stuck/{int(time.time())}.pkl'))

            n_warmstart = max(1, min(hp['warmstart'], int((1 - stuck_frac) * 5)))

            if needs_reset:
                self.mppi.reset()
                self.reset_trap_detection()
                n_warmstart = hp['warmstart']

            for k in range(n_warmstart):
                command, sub_time_s = self.mppi.command(phy, regrasp_goal, num_samples, viz=self.viz)
                self.mppi_viz(self.mppi, regrasp_goal, phy, command, sub_time_s)

            control_step(phy, command, sub_time_s)
            self.viz.viz(phy)

            if self.mov:
                self.mov.render(phy.d)

            results = regrasp_goal.get_results(phy)
            do_grasp_dynamics(phy, results)

            self.mppi.roll()

            itr += 1

    def check_is_stuck(self, phy):
        # TODO: split up q on a per-gripper basis
        latest_q = self.get_q_for_trap_detection(phy)
        self.state_history.insert(latest_q)
        qs = np.array(self.state_history.data)
        if self.state_history.full():
            # distance between the newest and oldest q in the buffer
            # the mean takes the average across joints.
            dq = (np.abs(qs[-1] - qs[0]) / len(self.state_history)).mean()
            self.max_dq = max(self.max_dq, dq)
            frac_dq = dq / self.max_dq
            rr.log_scalar('mab/frac_dq', frac_dq, color=[255, 0, 255])
            return frac_dq
        else:
            return 1

    def get_q_for_trap_detection(self, phy):
        return np.concatenate(
            (hp['q_joint_weight'] * phy.d.qpos[phy.o.rope.qpos_indices], phy.d.qpos[phy.o.robot.qpos_indices]))

    def mppi_viz(self, mppi, goal, phy, command, sub_time_s):
        if not self.viz.p.mppi_rollouts:
            return

        sorted_traj_indices = np.argsort(mppi.cost)

        i = None
        num_samples = mppi.cost.shape[0]
        for i in range(min(num_samples, 10)):
            sorted_traj_idx = sorted_traj_indices[i]
            cost_normalized = mppi.cost_normalized[sorted_traj_idx]
            c = list(cm.RdYlGn(1 - cost_normalized))
            c[-1] = 0.8
            result_i = mppi.rollout_results[:, sorted_traj_idx]
            goal.viz_result(phy, result_i, i, color=c, scale=0.002)
            rospy.sleep(0.001)  # needed otherwise messages get dropped :( I hate ROS...

        if command is not None:
            cmd_rollout_results, _, _ = rollout(phy.copy_all(), goal, np.expand_dims(command, 0),
                                                np.expand_dims(sub_time_s, 0), viz=None)
            goal.viz_result(phy, cmd_rollout_results, i, color='b', scale=0.004)

    def close(self):
        if self.mov is not None:
            self.mov.close()
