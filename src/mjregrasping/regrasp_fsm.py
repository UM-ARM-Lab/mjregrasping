from enum import Enum, auto

import numpy as np

from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner
from mjregrasping.params import hp
from mjregrasping.viz import Viz


class Modes(Enum):
    GOAL_SEEKING = auto()
    REGRASPING = auto()


TIMER_MAX = 50


class RegraspFSM:

    def __init__(self, gripper_names, current_locs, subgoals, viz: Viz):
        self.gripper_names = gripper_names
        self.mode = Modes.GOAL_SEEKING
        self.locs = current_locs.copy()
        self.subgoals = subgoals
        self.timer = TIMER_MAX
        self.viz = viz

    def update_state(self, phy, is_stuck, current_locs, planner: HomotopyRegraspPlanner):
        any_stuck = np.any(is_stuck)
        if self.mode == Modes.GOAL_SEEKING:
            if any_stuck:
                self.mode = Modes.REGRASPING
                print("Stuck! Replanning...")
                self.locs, self.subgoals, _, _ = planner.generate(phy, self.viz)
                self.timer = TIMER_MAX
                # planner.cost(np.array([1, 0]), phy.copy_all(), self.viz, viz_ik=True, log_loops=True,
                #              left_tool=0.553,
                #              right_tool=-1,
                #              left_tool_dx_1=0, left_tool_dy_1=0.03, left_tool_dz_1=0.04,
                #              left_tool_dx_2=-0.1, left_tool_dy_2=0.03, left_tool_dz_2=0.04,
                #              right_tool_dx_1=0, right_tool_dy_1=0.03, right_tool_dz_1=0.04,
                #              right_tool_dx_2=-0.1, right_tool_dy_2=0.03, right_tool_dz_2=0.04)
                return True
        elif self.mode == Modes.REGRASPING:
            self.timer -= 1
            if self.timer <= 0:
                print("Regrasp timed out! Replanning...")
                self.locs, self.subgoals, _, _ = planner.generate(phy, self.viz)
                self.timer = TIMER_MAX
                return True
            else:
                loc_is_close = abs(current_locs - self.locs) < hp['grasp_loc_diff_thresh']
                current_is_grasping = current_locs != -1
                regrasp_is_grasping = self.locs != -1
                reached = loc_is_close & (current_is_grasping == regrasp_is_grasping)
                all_reached = np.all(reached)
                if all_reached:
                    print("Regrasp complete!")
                    self.mode = Modes.GOAL_SEEKING
                    self.locs = current_locs.copy()
                    return True
        return False


class ComplexRegraspFSM:

    def __init__(self, gripper_names, current_locs, subgoals=None):
        self.gripper_names = gripper_names
        self.modes = [Modes.GOAL_SEEKING] * len(gripper_names)
        self.maintain_locs = current_locs
        self.regrasp_locs = current_locs.copy()
        self.subgoals = subgoals
        self.timer = TIMER_MAX

    def update_data(self, loc, subgoals):
        self.loc = loc
        self.subgoals = subgoals

    def update_state(self, is_stuck, current_locs, planner: HomotopyRegraspPlanner):
        # first check if any goal-seeking grippers are not grasping anything
        for mode, maintain_loc, regrasp_loc in zip(self.modes, self.maintain_locs, self.regrasp_locs):
            if mode == Modes.GOAL_SEEKING:
                if maintain_loc == -1:
                    pass
                    # plan conditioned on not changing and gripper that are currently goal-seeking AND grasping
                    # locs, subgoals, _, _ = planner.generate()
                    # set the mode to be REGRASPING
                    # update the regrasp_locs
            pass
        return False
        # current_is_grasping = current_locs != -1
        # regrasp_is_grasping = self.fsm.regrasp_locs != -1
        # loc_is_close = abs(current_locs - self.fsm.regrasp_locs) < hp['grasp_loc_diff_thresh']
        # regrasp_complete = loc_is_close & (current_is_grasping == regrasp_is_grasping)
        if self.state == Modes.GOAL_SEEKING:
            if is_stuck:
                print(f"{self.name} is trapped!")
                self.state = Modes.REGRASPING
                return True
            # if not is_grasping:
            #     print(f"{self.name} is not grasping while goal seeking. Using {regrasp_loc=}")
            #     self.state = RegraspStates.REGRASPING
            #     self.loc = regrasp_loc
            #     return True
        elif self.state == Modes.REGRASPING:
            if regrasp_completed:
                print(f"{self.name} regrasp completed!")
                self.state = Modes.GOAL_SEEKING
                return False
                # # If current_loc is -1, then the gripper is not grasping, so we don't change to GOAL_SEEKING.
                # if current_loc != -1:
                #     print(f"{self.name} regrasp completed!")
                #     self.state = RegraspStates.GOAL_SEEKING
                #     return True
        return False

    def __str__(self):
        return f"{self.name} is {self.state.name} at {self.loc:.3f}"

    def __repr__(self):
        return str(self)
