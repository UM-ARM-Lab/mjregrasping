import time
from pathlib import Path

import mujoco.viewer
import numpy as np

from mjregrasping.grasping import activate_grasp
from mjregrasping.mjsaver import save_data_and_eq
from mjregrasping.physics import Physics


def main():
    m = mujoco.MjModel.from_xml_path('models/untangle_scene.xml')
    d = mujoco.MjData(m)
    phy = Physics(m, d, "computer_rack")

    q = np.array([
        -0.5, 0.4,  # torso
        -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
        0.1,  # left gripper
        1.2, -0.2, 0, -0.90, 0, -0.6, 0,  # right arm
        0.3,  # right gripper
    ])
    d.qpos[:18] = q
    d.act[:18] = q

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(m, d)

            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Useful lines of code to run
    activate_grasp(phy, 'right', 0.75, phy.o.rope.body_indices)
    activate_grasp(phy, 'left', 0.45, phy.o.rope.body_indices)
    phy.m.eq("right").active = 0
    phy.m.eq("left").active = 0
    now = int(time.time())
    save_data_and_eq(phy, Path(f"states/untangle/{now}.pkl"))


if __name__ == '__main__':
    main()
