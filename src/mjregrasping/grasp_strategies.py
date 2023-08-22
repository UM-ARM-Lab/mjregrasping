from enum import Enum, auto


class Strategies(Enum):
    NEW_GRASP = auto()
    RELEASE = auto()
    MOVE = auto()
    STAY = auto()


def get_strategy(current_locs, next_locs):
    strategy = []
    for current_loc_i, next_loc_i in zip(current_locs, next_locs):
        if current_loc_i == -1 and next_loc_i != -1:
            strategy.append(Strategies.NEW_GRASP)
        elif current_loc_i == next_loc_i:
            strategy.append(Strategies.STAY)
        elif current_loc_i != -1 and next_loc_i == -1:
            strategy.append(Strategies.RELEASE)
        elif current_loc_i != next_loc_i:
            strategy.append(Strategies.MOVE)
    return strategy
