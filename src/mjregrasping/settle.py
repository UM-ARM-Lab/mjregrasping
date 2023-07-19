from typing import Optional, Callable

import numpy as np

from mjregrasping.movie import MjMovieMaker
from mjregrasping.rollout import control_step, no_results
from mjregrasping.viz import Viz


def settle(phy, sub_time_s, viz: Optional[Viz], is_planning, settle_steps=20, mov: Optional[MjMovieMaker] = None,
           result_func: Optional[Callable] = no_results):
    ctrl = np.zeros(phy.m.nu)

    results = []
    for _ in range(settle_steps):
        if viz:
            viz.viz(phy, is_planning)
        control_step(phy, ctrl, sub_time_s=sub_time_s)

        if not is_planning and mov:
            mov.render(phy.d)

        result_tuple = result_func(phy)

        results.append(result_tuple)

    return results
