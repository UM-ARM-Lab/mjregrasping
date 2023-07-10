import time
from pathlib import Path

import mujoco.viewer

import rospy
from arc_utilities import ros_init
from mjregrasping.grasping import activate_grasp
from mjregrasping.mjsaver import save_data_and_eq
from mjregrasping.mujoco_objects import Objects
from mjregrasping.physics import Physics
from mjregrasping.scenarios import conq_hose, setup_conq_hose_scene
from mjregrasping.viz import make_viz
from std_msgs.msg import String


@ros_init.with_ros("viewer")
def main():
    scenario = conq_hose
    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))

    d = mujoco.MjData(m)
    phy = Physics(m, d, objects=Objects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))
    viz = make_viz(scenario)

    setup_conq_hose_scene(phy, viz)

    latest_cmd = ""

    def cmd_callback(msg):
        nonlocal latest_cmd
        latest_cmd = msg.data

    cmd_sub = rospy.Subscriber("viewer_cmd", String, queue_size=10, callback=cmd_callback)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(m, d)

            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            if latest_cmd == "save":
                now = int(time.time())
                path = Path(f"states/conq_hose/{now}.pkl")
                print(f"Saving to {path}")
                save_data_and_eq(phy, path)
                latest_cmd = ""
            elif "grasp" in latest_cmd:
                _, eq_name, loc = latest_cmd.split(" ")
                loc = float(loc)
                latest_cmd = ""
                activate_grasp(phy, eq_name, loc)
            elif "release" in latest_cmd:
                _, eq_name = latest_cmd.split(" ")
                phy.m.eq(eq_name).active = 0
                latest_cmd = ""


if __name__ == '__main__':
    main()
