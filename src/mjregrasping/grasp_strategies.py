from enum import Enum, auto


class Strategies(Enum):
    NEW_GRASP = auto()
    RELEASE = auto()
    MOVE = auto()
    STAY = auto()
