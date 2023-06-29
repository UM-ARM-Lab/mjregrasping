from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import rerun as rr
from matplotlib import cm

import rospy
from mjregrasping.buffer import Buffer
from mjregrasping.goals import RegraspGoal
from mjregrasping.movie import MjMovieMaker
from mjregrasping.params import hp
from mjregrasping.regrasping_mppi import RegraspMPPI, do_grasp_dynamics, regrasp_rollout
from mjregrasping.rollout import control_step
from mjregrasping.settle import settle
from mjregrasping.viz import Viz


class RegraspMPC:

    def __init__(self, pool: ThreadPoolExecutor, mppi_nu: int, skeletons, goal, seed: int, viz: Viz,
                 mov: Optional[MjMovieMaker] = None):
        self.mppi_nu = mppi_nu
        self.skeletons = skeletons
        self.pool = pool
        self.viz = viz
        self.op_goal = goal
        self.n_g = hp['n_g']
        self.mov = mov
        self.is_gasping_rng = np.random.RandomState(0)

        self.mppi = RegraspMPPI(pool=self.pool, nu=self.mppi_nu, seed=seed, horizon=hp['regrasp_horizon'],
                                noise_sigma=np.deg2rad(2),
                                n_g=self.n_g, temp=hp['regrasp_temp'])
        self.state_history = Buffer(hp['state_history_size'])
        self.max_dq = None
        self.reset_trap_detection()

    def reset_trap_detection(self):
        self.state_history.reset()
        self.max_dq = 0

    def run(self, phy, attach_pos):
        num_samples = hp['regrasp_n_samples']

        regrasp_goal = RegraspGoal(self.op_goal, self.skeletons, hp['grasp_goal_radius'], self.viz)
        regrasp_goal.recompute_candidates(phy, attach_pos)

        self.mppi.reset()
        self.reset_trap_detection()

        itr = 0
        max_iters = 5000
        sub_time_s = hp['sub_time_s']
        warmstart_count = 0
        self.viz.viz(phy)
        while True:
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown")

            if itr > max_iters:
                return False

            regrasp_goal.viz_goal(phy)
            if regrasp_goal.satisfied(phy):
                print("Goal reached!")
                return True

            if self.get_mab_reward(phy) < 0.2 and itr % 10 == 0:
                regrasp_goal.recompute_candidates(phy, attach_pos)
                self.reset_trap_detection()

            while warmstart_count < hp['warmstart']:
                command = self.mppi.command(phy, regrasp_goal, sub_time_s, num_samples, viz=self.viz)
                self.mppi_viz(self.mppi, regrasp_goal, phy, command, sub_time_s)
                warmstart_count += 1

            command = self.mppi.command(phy, regrasp_goal, sub_time_s, num_samples, viz=self.viz)
            self.mppi_viz(self.mppi, regrasp_goal, phy, command, sub_time_s)

            control_step(phy, command, sub_time_s)
            self.viz.viz(phy)

            if self.mov:
                self.mov.render(phy.d)

            results = regrasp_goal.get_results(phy)
            did_new_grasp = do_grasp_dynamics(phy, results)
            if did_new_grasp:
                print("New grasp!")
                settle(phy, sub_time_s, self.viz, is_planning=False)
                warmstart_count = 0
                self.mppi.reset()

            self.mppi.roll()

            itr += 1

    def get_mab_reward(self, phy):
        latest_q = self.get_q_for_trap_detection(phy)
        self.state_history.insert(latest_q)
        qs = np.array(self.state_history.data)
        if self.state_history.full():
            # distance between the newest and oldest q in the buffer
            # the mean takes the average across joints.
            dq = (np.abs(qs[-1] - qs[0]) / len(self.state_history)).mean()
            self.max_dq = max(self.max_dq, dq)
            frac_dq = dq / self.max_dq
            rr.log_scalar('mab/dq', dq, color=[0, 255, 0])
            rr.log_scalar('mab/max_dq', self.max_dq, color=[0, 0, 255])
            rr.log_scalar('mab/frac_dq', frac_dq, color=[255, 0, 255])
            return frac_dq
        else:
            return np.inf

    def get_q_for_trap_detection(self, phy):
        return np.concatenate(
            (hp['q_joint_weight'] * phy.d.qpos[phy.o.rope.qpos_indices], phy.d.qpos[phy.o.val.qpos_indices]))

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
            goal.viz_result(result_i, i, color=c, scale=0.002)
            rospy.sleep(0.001)  # needed otherwise messages get dropped :( I hate ROS...

        if command is not None:
            cmd_rollout_results, _, _ = regrasp_rollout(phy.copy_all(), goal, sub_time_s, command[None], viz=None)
            goal.viz_result(cmd_rollout_results, i, color='b', scale=0.004)

    def close(self):
        if self.mov is not None:
            self.mov.close()
