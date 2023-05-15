from dataclasses import dataclass

import mujoco


@dataclass
class Physics:
    m: mujoco.MjModel
    d: mujoco.MjData
