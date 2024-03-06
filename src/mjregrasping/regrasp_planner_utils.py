import itertools
from dataclasses import dataclass
from typing import List

import numpy as np

from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_is_grasping
from mjregrasping.physics import Physics
from moveit_msgs.msg import MotionPlanResponse


def get_geodesic_dist(locs, key_loc: float):
    return np.abs(locs - key_loc) @ (locs != -1)


def get_will_be_grasping(s: Strategies, is_grasping: bool):
    if is_grasping:
        return s in [Strategies.STAY, Strategies.MOVE]
    else:
        return s in [Strategies.NEW_GRASP]


def get_all_strategies_from_phy(phy: Physics):
    is_grasping = get_is_grasping(phy)
    strategies_per_gripper = []
    for i, is_grasping_i in enumerate(is_grasping):
        strategies = []
        for strategy in Strategies:
            if strategy == Strategies.NEW_GRASP:
                if is_grasping_i:
                    continue  # not possible
                else:
                    strategies.append(strategy)
            elif strategy == Strategies.RELEASE:
                if not is_grasping_i:
                    continue  # not possible
                else:
                    strategies.append(strategy)
            elif strategy == Strategies.MOVE:
                if not is_grasping_i:
                    continue  # not possible
                else:
                    strategies.append(strategy)
            elif strategy == Strategies.STAY:
                strategies.append(strategy)
            else:
                raise NotImplementedError(strategy)
        strategies_per_gripper.append(strategies)

    all_strategies = list(itertools.product(*strategies_per_gripper))

    all_strategies = np.array(all_strategies)
    return all_strategies


@dataclass
class SimGraspCandidate:
    phy0: Physics
    phy: Physics
    strategy: List[Strategies]
    res: MotionPlanResponse
    locs: np.ndarray
    initial_locs: np.ndarray


@dataclass
class SimGraspInput:
    strategy: List[Strategies]
    candidate_locs: np.ndarray
