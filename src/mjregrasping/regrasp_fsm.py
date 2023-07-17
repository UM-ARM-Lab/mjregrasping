from enum import Enum, auto


class RegraspStates(Enum):
    GOAL_SEEKING = auto()
    REGRASPING = auto()


class RegraspFSM:

    def __init__(self, name: str, current_loc):
        self.state = RegraspStates.GOAL_SEEKING
        self.name = name
        self.loc = current_loc

    def update(self, is_grasping: bool, is_stuck: bool, regrasp_completed: bool, current_loc: float,
               regrasp_loc: float):
        """
        Args:
            is_grasping: whether the gripper is currently grasping
            is_stuck: the current state of the trap detection
            regrasp_completed: whether a regrasp is now completed
            current_loc: the current location of the gripper, -1 for not grasping, or [0-1] for grasping
            regrasp_loc: the desired location of the regrasp

        Returns: True if the state has changed

        """
        if self.state == RegraspStates.GOAL_SEEKING:
            if is_stuck:
                print(f"{self.name} is trapped!")
                self.state = RegraspStates.REGRASPING
                self.loc = regrasp_loc
                return True
            if not is_grasping:
                print(f"{self.name} is not grasping while goal seeking. Using {regrasp_loc=}")
                self.state = RegraspStates.REGRASPING
                self.loc = regrasp_loc
                return True
        elif self.state == RegraspStates.REGRASPING:
            if regrasp_completed:
                print(f"{self.name} regrasp completed!")
                self.loc = current_loc
                if current_loc != -1:
                    # If current_loc is -1, then the gripper is not grasping, so we don't change to GOAL_SEEKING.
                    self.state = RegraspStates.GOAL_SEEKING
                return True
        return False

    def __str__(self):
        return f"{self.name} is {self.state.name} at {self.loc:.3f}"

    def __repr__(self):
        return str(self)
