from typing import Optional, Callable

import numpy as np

from mjregrasping.movie import MjMovieMaker
from mjregrasping.rollout import control_step, get_result_tuple
from mjregrasping.viz import Viz


def settle(phy, sub_time_s, viz: Viz, is_planning, settle_steps=50, mov: Optional[MjMovieMaker] = None,
           ctrl: Optional[np.ndarray] = None, get_result_func: Optional[Callable] = None,
           policy: Optional[Callable] = None):
    if ctrl is None:
        ctrl = np.zeros(phy.m.nu)

    results_lists = None
    for _ in range(settle_steps):
        viz.viz(phy, is_planning)
        if policy is not None:
            ctrl = policy(phy)
        control_step(phy, ctrl, sub_time_s=sub_time_s)

        if get_result_func is not None:
            result_tuple = get_result_tuple(get_result_func, phy)

            if results_lists is None:
                results_lists = tuple([] for _ in result_tuple)

            for result_list, result in zip(results_lists, result_tuple):
                result_list.append(result)

        if not is_planning and mov:
            mov.render(phy.d)

    if results_lists is None:
        return None

    return tuple(np.array(result_i) for result_i in results_lists)
