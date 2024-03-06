#!/usr/bin/env python3
from itertools import cycle

import mujoco
import numpy as np
from vedo import Line, Text2D, DashedLine

import rospy
from mjregrasping.drone_homotopy import get_tree_skeletons, get_h_for_drones
from mjregrasping.mjvedo import MjVedo
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
    m.body("drone1").pos[0] = 0
    m.body("drone1").pos[1] = -4
    m.body("drone1").pos[2] = 2

    objects = MjObjects(m, 'obstacles', drones, "rope")
    phy = Physics(m, mujoco.MjData(m), objects)
    mujoco.mj_forward(phy.m, phy.d)
    graph, h, loops = get_h_for_drones(phy, skeletons)
    viz_h(xml_path, phy, skeletons, graph, h, loops, 'release')

    # Grasp the end, on the other side of the tree
    m.eq('drone1').active = True
    m.body("drone1").pos[0] = 0
    m.body("drone1").pos[1] = -14.4
    m.body("drone1").pos[2] = -2
    m.eq("drone1").obj2id = phy.m.body("B_1").id

    objects = MjObjects(m, 'obstacles', drones, "rope")
    phy = Physics(m, mujoco.MjData(m), objects)
    mujoco.mj_forward(phy.m, phy.d)
    graph, h, loops = get_h_for_drones(phy, skeletons)

    viz_h(xml_path, phy, skeletons, graph, h, loops, 'grasp')


def viz_h(xml_path, phy, skeletons, graph, h, loops, name):
    mjvedo = MjVedo(xml_path)

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
    for _, skel in skeletons.items():
        skel_line = Line(skel, lw=25, alpha=1)  # initially these will not be shown
        skel_lines.append(skel_line)
        mjvedo.plotter += skel_line
    loop_lines = []

    for loop, c in zip(loops, cycle(colors)):
        loop_line = DashedLine(loop, lw=25, alpha=1, c=c, spacing=0.4)
        loop_lines.append(loop_line)
        mjvedo.plotter += loop_line

    # render the scene and fill the actor_map, which lets us reference mujoco bodies by name for animation
    mjvedo.viz(phy)

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

    tree.alpha(0.4)

    mjvedo.plotter.camera.SetViewUp(0, 0, 1)
    mjvedo.plotter.camera.SetFocalPoint(2, -10, 2)
    mjvedo.plotter.camera.SetPosition(-4, -8, 4)
    mjvedo.plotter.show()


    mjvedo.plotter.render().screenshot(f"results/drone_example_{name}.png", 3)


if __name__ == "__main__":
    main()
