from time import sleep
import mujoco.viewer
import mujoco


def main():
    m = mujoco.MjModel.from_xml_path("dynamic_grasp_demo.xml")
    d = mujoco.MjData(m)

    T = 50

    with mujoco.viewer.launch_passive(m, d) as v:
        # move forward towards the rope
        d.ctrl[1] = 0.2
        for i in range(T):
            v.sync()

            mujoco.mj_step(m, d, nstep=10)
            sleep(0.05)

        # grasp the rope
        eq = m.eq("grasp")
        print(eq)  # print some info
        m.eq_data[0, 3:6] = 0.0  # zero the position offset between the two bodies
        eq.active[:] = 1.0

        # move the rope to the side
        d.ctrl[0] = -0.2
        for i in range(T):
            v.sync()

            mujoco.mj_step(m, d, nstep=10)
            sleep(0.05)


if __name__ == '__main__':
    main()
