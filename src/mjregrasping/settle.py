from typing import Optional, Callable, Tuple

import numpy as np

from mjregrasping.movie import MjMovieMaker
from mjregrasping.rollout import control_step, get_result_tuple, no_results, list_of_tuples_to_tuple_of_arrays
from mjregrasping.viz import Viz


def settle(phy, sub_time_s, viz: Optional[Viz], is_planning, settle_steps=20, mov: Optional[MjMovieMaker] = None,
           ctrl: Optional[np.ndarray] = None, get_result_func: Optional[Callable] = no_results,
           policy: Optional[Callable] = None):
    if ctrl is None:
        ctrl = np.zeros(phy.m.nu)

    results = []
    for _ in range(settle_steps):
        if viz:
            viz.viz(phy, is_planning)
        if policy is not None:
            ctrl = policy(phy)
        control_step(phy, ctrl, sub_time_s=sub_time_s)

        if not is_planning and mov:
            mov.render(phy.d)

        result_tuple: Tuple = get_result_tuple(get_result_func, phy)

        results.append(result_tuple)

    return list_of_tuples_to_tuple_of_arrays(results)
