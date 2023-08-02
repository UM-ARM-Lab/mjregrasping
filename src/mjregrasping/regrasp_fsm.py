import time
from enum import Enum, auto
from pathlib import Path

import numpy as np

from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner, grasp_and_settle, release_and_settle, \
    execute_grasp_plan
from mjregrasping.mjsaver import save_data_and_eq
from mjregrasping.physics import Physics
from mjregrasping.rrt import GraspRRT
from mjregrasping.viz import Viz
from moveit_msgs.msg import MoveItErrorCodes
from trajectory_msgs.msg import JointTrajectory


class Modes(Enum):
    GOAL_SEEKING = auto()
    REGRASPING = auto()


TIMER_MAX = 50


class RegraspFSM:

    def __init__(self, grasp_rrt: GraspRRT, gripper_names, current_locs, subgoals, viz: Viz):
        self.grasp_joint_trajectory = JointTrajectory()
        self.gripper_names = gripper_names
        self.mode = Modes.GOAL_SEEKING
        self.locs = current_locs.copy()
        self.subgoals = subgoals
        self.timer = TIMER_MAX
        self.viz = viz
        self.grasp_rrt = grasp_rrt

    def update_state(self, phy: Physics, is_stuck, current_locs, planner: HomotopyRegraspPlanner):
        any_stuck = np.any(is_stuck)
        if self.mode == Modes.GOAL_SEEKING:
            if any_stuck:
                # self.mode = Modes.REGRASPING  # NOTE: currently not using "states" properly
                print("Stuck! Replanning...")
                self.locs, self.subgoals, strategy = planner.generate(phy, self.viz)

                # now execute the plan
                release_and_settle(phy, strategy, self.viz, is_planning=False)
                # Since it's a pain to get the previously computed plan out of the bayes-opt planner, just recompute it
                for i in range(10):
                    res = self.grasp_rrt.plan(phy.copy_all(), strategy, self.locs, self.viz)
                    if res.error_code.val == MoveItErrorCodes.SUCCESS:
                        break
                    self.grasp_rrt.display_result(res)
                else:
                    raise RuntimeError("Failed to plan!")
                # planner.cost(strategy, phy, self.viz,
                #              left_tool=0.231,
                #              right_tool=-1,
                #              left_tool_dx_1=0, left_tool_dy_1=0.03, left_tool_dz_1=0.04,
                #              left_tool_dx_2=-0.1, left_tool_dy_2=0.03, left_tool_dz_2=0.04,
                #              right_tool_dx_1=0, right_tool_dy_1=0.03, right_tool_dz_1=0.04,
                #              right_tool_dx_2=-0.1, right_tool_dy_2=0.03, right_tool_dz_2=0.04)
                #
                execute_grasp_plan(phy, res, self.viz, is_planning=False)
                grasp_and_settle(phy, self.locs, self.viz, is_planning=False)

                save_data_and_eq(phy, Path(f'states/on_stuck/{int(time.time())}.pkl'))
                self.timer = TIMER_MAX
                return True
        # elif self.mode == Modes.REGRASPING:
        #     self.timer -= 1
        #     if self.timer <= 0:
        #         print("Regrasp timed out! Replanning...")
        #         save_data_and_eq(phy, Path(f'states/on_timeout/{int(time.time())}.pkl'))
        #         self.locs, self.subgoals, strategy = planner.generate(phy, self.viz)
        #         self.timer = TIMER_MAX
        #         return True
        #     else:
        #         loc_is_close = abs(current_locs - self.locs) < hp['grasp_loc_diff_thresh']
        #         current_is_grasping = current_locs != -1
        #         regrasp_is_grasping = self.locs != -1
        #         reached = loc_is_close & (current_is_grasping == regrasp_is_grasping)
        #         all_reached = np.all(reached)
        #         if all_reached:
        #             print("Regrasp complete!")
        #             self.mode = Modes.GOAL_SEEKING
        #             self.locs = current_locs.copy()
        #             return True
        return False
