#!/usr/bin/env python3
import mujoco
import numpy as np
import rerun as rr
from mujoco.viewer import launch_passive

from mjregrasping.grasping import activate_grasp
from mjregrasping.homotopy_checker import get_full_gl_signature_from_phy
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun
from mjregrasping.rope_gripper_collision import disable_rope_gripper_collisions
from mjregrasping.scenarios import val_untangle, get_untangle_skeletons


def main():
    rr.init("robot_grasping_cable_demo")
    rr.spawn()

    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    scenario = val_untangle

    gl_ctx = mujoco.GLContext(1280, 720)
    gl_ctx.make_current()

    mjrr = MjReRun(xml_path=scenario.xml_path)

    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
    d = mujoco.MjData(m)
    o = MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
    phy = Physics(m, d, o)

    skeletons = get_untangle_skeletons(phy)

    disable_rope_gripper_collisions(phy)

    # Move the rope to the attach point
    rope_xyz_q_indices = phy.o.rope.qpos_indices[:3]
    phy.d.qpos[rope_xyz_q_indices] = phy.d.body("attach").xpos
    phy.m.eq("attach").data[3:6] = 0

    # Start the mujoco passive viewer
    grasped = False

    def compute_gl_sig_and_viz():
        h, grasp_loops = get_full_gl_signature_from_phy(skeletons, phy, False, False)
        print(f'Homotopy: {h}')

        mjrr.viz(phy)

        for i, grasp_loop in enumerate(grasp_loops):
            rr.log(f'grasp_loops/{i}', rr.LineStrips3D(grasp_loop))

        for k, obstacle_loop in skeletons.items():
            rr.log(f'skeleton/{k}', rr.LineStrips3D(obstacle_loop))

    def _key_cb(key):
        if key == ord('h'):
            compute_gl_sig_and_viz()

    with mujoco.viewer.launch_passive(m, d, key_callback=_key_cb) as viewer:
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()

            # first settle, then grasp
            if not grasped:
                activate_grasp(phy, 'left', 0.50)
                activate_grasp(phy, 'right', 1.00)
                grasped = True

            if d.time > 20.0:
                compute_gl_sig_and_viz()

                print("Press 'h' to re-compute the homotopy signature")


if __name__ == "__main__":
    main()
