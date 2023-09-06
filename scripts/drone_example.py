#!/usr/bin/env python3
from itertools import cycle

import mujoco
import numpy as np
from vedo import Line, Text2D

import rospy
from mjregrasping.drone_homotopy import get_tree_skeletons, get_h_for_drones
from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.homotopy_checker import get_full_h_signature, create_graph_nodes
from mjregrasping.mjvedo import MjVedo
from mjregrasping.mujoco_object import MjObject
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.physics import Physics
from mjregrasping.robot_data import drones


def main():
    rospy.init_node("drone_example")

    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    xml_path = 'models/drone_scene.xml'

    m = mujoco.MjModel.from_xml_path(xml_path)

    objects = MjObjects(m, 'obstacles', drones, "rope")
    phy = Physics(m, mujoco.MjData(m), objects)
    mujoco.mj_forward(phy.m, phy.d)
    skeletons = get_tree_skeletons(phy)
    graph, h, loops = get_h_for_drones(phy, skeletons)
    viz_h(xml_path, phy, skeletons, graph, h, loops, 'initial')

    # Release and move away from the pipe
    m.eq('drone1').active = False
    m.body("drone1").pos[0] = -5

    objects = MjObjects(m, 'obstacles', drones, "rope")
    phy = Physics(m, mujoco.MjData(m), objects)
    mujoco.mj_forward(phy.m, phy.d)
    graph, h, loops = get_h_for_drones(phy, skeletons)
    viz_h(xml_path, phy, skeletons, graph, h, loops, 'release')

    # Grasp the end, on the other side of the tree
    m.eq('drone1').active = True
    m.body("drone1").pos[0] = 0
    m.body("drone1").pos[1] = -14.4
    m.eq("drone1").obj2id = phy.m.body("B_1").id

    objects = MjObjects(m, 'obstacles', drones, "rope")
    phy = Physics(m, mujoco.MjData(m), objects)
    mujoco.mj_forward(phy.m, phy.d)
    graph, h, loops = get_h_for_drones(phy, skeletons)

    viz_h(xml_path, phy, skeletons, graph, h, loops, 'grasp')


def viz_h(xml_path, phy, skeletons, graph, h, loops, name):
    mjvedo = MjVedo(xml_path)
    print(h)

    mjvedo.plotter += Text2D(f"Example: Carrying a Draining Tube with Drones \n{h=}", 'top-center')

    colors = [
        'red',
        'green',
        'blue',
        'orange',
        'purple',
        'brown',
        'pink',
    ]

    skel_lines = []
    for name, skel in skeletons.items():
        skel_line = Line(skel, lw=3, alpha=0)  # initially these will not be shown
        skel_lines.append(skel_line)
        mjvedo.plotter += skel_line
    loop_lines = []

    for loop, c in zip(loops, cycle(colors)):
        loop_line = Line(loop, lw=3, alpha=0, c=c)
        loop_lines.append(loop_line)
        mjvedo.plotter += loop_line

    # render the scene and fill the actor_map, which lets us reference mujoco bodies by name for animation
    mjvedo.viz(phy)

    n_spins = 2
    seconds_per_spin = 7
    num_frames = mjvedo.num_frames_for_spin(seconds_per_spin, n_spins)
    total_rotation = n_spins * 2 * np.pi
    frames_per_spin = num_frames / n_spins
    cx = 2
    cy = -10
    distance = 10
    z = 10

    tree = mjvedo.get_actor(phy, 'tree')
    drone_actors = []
    for geom_id in range(phy.m.ngeom):
        geom_name = phy.m.geom(geom_id).name
        if 'drone' in geom_name:
            actor = mjvedo.get_actor(phy, geom_name)
            if isinstance(actor, list):
                drone_actors.extend(actor)
            else:
                drone_actors.append(actor)
    rope_actors = []
    for rope_geom_name in phy.o.rope.geom_names:
        actor = mjvedo.get_actor(phy, rope_geom_name)
        if isinstance(actor, list):
            rope_actors.extend(actor)
        else:
            rope_actors.append(actor)

    def anim(t, plotter):
        spin_idx = int(t / frames_per_spin)

        azimuth = t * total_rotation / num_frames

        x = distance * np.cos(azimuth) + cx
        y = distance * np.sin(azimuth) + cy
        plotter.camera.SetPosition(x, y, z)
        plotter.camera.SetFocalPoint(cx, cy, 0)
        plotter.camera.SetViewUp(0, 0, 1)

        if t == 1:
            mjvedo.plotter.screenshot("results/drone_example.png")

        if spin_idx == 1:
            disappear_alpha = np.exp(-8 * (t % frames_per_spin) / frames_per_spin)
            tree.alpha(disappear_alpha)
            for drone in drone_actors:
                drone.alpha(disappear_alpha)
            for rope_actor in rope_actors:
                rope_actor.alpha(disappear_alpha)
            appear_alpha = 1 - disappear_alpha
            for skel_line in skel_lines:
                skel_line.alpha(appear_alpha)
            for loop_line in loop_lines:
                loop_line.alpha(appear_alpha)


    mjvedo.record(f"results/drone_example_{name}.mp4", num_frames=num_frames, anim_func=anim)


    # plt.figure()
    # nx.draw(graph, with_labels=True, font_weight='bold')
    # plt.show()


if __name__ == "__main__":
    main()
