import numpy as np
import rerun as rr

from mjregrasping.buffer import Buffer
from mjregrasping.eq_errors import compute_total_eq_error
from mjregrasping.goals import GraspLocsGoal
from mjregrasping.params import hp
from mjregrasping.physics import Physics


def get_q_for_trap_detection(phy):
    return np.concatenate(
        (hp['trap_q_rope_weight'] * phy.d.qpos[phy.o.rope.qpos_indices], phy.d.qpos[phy.o.robot.qpos_indices]))


class TrapDetection:

    def __init__(self):
        self.state_history = Buffer(hp['state_history_size'])
        self.frac_dq = 1
        self.max_dq = 0
        self.reset_trap_detection()

    def reset_trap_detection(self):
        self.state_history.reset()
        self.max_dq = 0

    def check_is_stuck(self, phy: Physics, grasp_goal: GraspLocsGoal):
        not_moving = self.check_is_not_moving(phy)
        no_grasp_goals = np.all(grasp_goal.get_grasp_locs() == -1)
        return not_moving or no_grasp_goals

    def check_is_not_moving(self, phy: Physics):
        latest_q = get_q_for_trap_detection(phy)
        self.state_history.insert(latest_q)
        qs = np.array(self.state_history.data)
        if self.state_history.full():
            # distance between the newest and oldest q in the buffer
            # the mean takes the average across joints.
            dq = (np.abs(qs[-1] - qs[0]) / len(self.state_history)).mean()
            # taking min with max_max_dq means if we moved a really large amount, we cap it so that our
            # trap detection isn't thrown off.
            self.max_dq = min(max(self.max_dq, dq), hp['max_max_dq'])
            self.frac_dq = min(dq / self.max_dq, 1)
            rr.log_scalar('trap_detection/frac_dq', self.frac_dq, color=[255, 0, 255])
            rr.log_scalar('trap_detection/threshold', hp['frac_dq_threshold'], color=[255, 0, 0])

            return self.frac_dq < hp['frac_dq_threshold']
        else:
            rr.log_scalar('trap_detection/frac_dq', 1, color=[255, 0, 255])
            rr.log_scalar('trap_detection/threshold', hp['frac_dq_threshold'], color=[255, 0, 0])
            return False
