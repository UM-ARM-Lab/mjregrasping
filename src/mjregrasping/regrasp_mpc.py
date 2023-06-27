import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import rerun as rr
from matplotlib import cm

import rospy
from mjregrasping.body_with_children import Objects
from mjregrasping.buffer import Buffer
from mjregrasping.goals import RegraspGoal
from mjregrasping.movie import MjMovieMaker
from mjregrasping.mujoco_mppi import MujocoMPPI
from mjregrasping.params import hp
from mjregrasping.regrasping_mppi import RegraspMPPI, do_grasp_dynamics
from mjregrasping.rollout import control_step, rollout
from mjregrasping.settle import settle
from mjregrasping.viz import Viz

logger = logging.getLogger(f'rosout.{__name__}')


class RegraspMPC:

    def __init__(self, mppi_nu: int, pool: ThreadPoolExecutor, viz: Viz, goal, objects: Objects,
                 seed: int = 1, mov: Optional[MjMovieMaker] = None):
        self.mppi_nu = mppi_nu
        self.pool = pool
        self.viz = viz
        self.op_goal = goal
        self.objects = objects
        self.rope_body_indices = self.objects.rope.body_indices
        self.n_g = hp['n_g']
        self.mov = mov
        self.is_gasping_rng = np.random.RandomState(0)

        self.mppi = MujocoMPPI(pool=self.pool, nu=mppi_nu, seed=seed, noise_sigma=np.deg2rad(2), horizon=hp['horizon'],
                               temp=hp['temp'])
        self.state_history = Buffer(hp['state_history_size'])
        self.max_dq = None
        self.reset_trap_detection()

    def reset_trap_detection(self):
        self.state_history.reset()
        self.max_dq = 0

    def run(self, phy):

        # copy model and data since each solution should be different/independent
        num_samples = hp['regrasp_n_samples']

        regrasp_goal = RegraspGoal(self.op_goal, hp['grasp_goal_radius'], self.objects, self.viz)

        # TODO: seed properly
        mppi = RegraspMPPI(pool=self.pool, nu=self.mppi_nu, seed=0, horizon=hp['regrasp_horizon'],
                           noise_sigma=np.deg2rad(2),
                           n_g=self.n_g, rope_body_indices=self.rope_body_indices, temp=hp['regrasp_temp'])
        mppi.reset()
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

            while warmstart_count < hp['warmstart']:
                command = mppi.command(phy, regrasp_goal, sub_time_s, num_samples, viz=self.viz)
                self.mppi_viz(mppi, regrasp_goal, phy, None, sub_time_s)
                warmstart_count += 1

            command = mppi.command(phy, regrasp_goal, sub_time_s, num_samples, viz=self.viz)
            self.mppi_viz(mppi, regrasp_goal, phy, None, sub_time_s)

            control_step(phy, command, sub_time_s)
            self.viz.viz(phy)

            if self.mov:
                self.mov.render(phy.d)

            results = regrasp_goal.get_results(phy)
            did_new_grasp = do_grasp_dynamics(phy, self.rope_body_indices, results)
            if did_new_grasp:
                print("New grasp!")
                settle(phy, sub_time_s, self.viz, is_planning=False, ctrl=phy.d.ctrl)
                self.reset_trap_detection()
                warmstart_count = 0
                mppi.reset()

            mppi.roll()

            itr += 1

    def check_needs_regrasp(self, data):
        latest_q = self.get_q_for_trap_detection(data)
        self.state_history.insert(latest_q)
        qs = np.array(self.state_history.data)
        if len(qs) < 2:
            dq = 0
        else:
            # distance between the newest and oldest q in the buffer
            # the mean takes the average across joints.
            dq = (np.abs(qs[-1] - qs[0]) / len(self.state_history)).mean()
        self.max_dq = max(self.max_dq, dq)
        has_not_moved = dq < hp['frac_max_dq'] * self.max_dq
        needs_regrasp = self.state_history.full() and has_not_moved
        rr.log_scalar('needs_regrasp/dq', dq, color=[0, 255, 0])
        rr.log_scalar('needs_regrasp/max_dq', self.max_dq, color=[0, 0, 255])
        rr.log_scalar('needs_regrasp/dq_threshold', self.max_dq * hp['frac_max_dq'], color=[255, 0, 0])
        return needs_regrasp

    def get_q_for_trap_detection(self, data):
        return np.concatenate((hp['q_joint_weight'] * data.qpos[self.objects.rope.qpos_indices],
                               data.qpos[self.objects.val.qpos_indices]))

    def mppi_viz(self, mppi, goal, phy, command, sub_time_s):
        if not self.viz.p.mppi_rollouts:
            return

        sorted_traj_indices = np.argsort(mppi.cost)

        i = None
        num_samples = mppi.cost.shape[0]
        for i in range(min(num_samples, 10)):
            sorted_traj_idx = sorted_traj_indices[i]
            cost_normalized = mppi.cost_normalized[sorted_traj_idx]
            c = cm.RdYlGn(1 - cost_normalized)
            result_i = mppi.rollout_results[:, sorted_traj_idx]
            goal.viz_result(result_i, i, color=c, scale=0.002)
            rospy.sleep(0.01)  # needed otherwise messages get dropped :( I hate ROS...

        if command is not None:
            cmd_rollout_results = rollout(phy.copy_data(), command[None], sub_time_s, get_result_func=goal.get_results)
            goal.viz_result(cmd_rollout_results, i, color='b', scale=0.004)

    def close(self):
        if self.mov is not None:
            self.mov.close()

    def sample_is_grasping(self, param):
        r = self.is_gasping_rng.uniform(size=self.n_g)
        is_grasping = (r < param).astype(int)
        return is_grasping
