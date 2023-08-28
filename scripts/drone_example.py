#!/usr/bin/env python3
import matplotlib.pyplot as plt
import mujoco
import networkx as nx
import numpy as np
import rerun as rr

import rospy
from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.homotopy_checker import get_full_h_signature, create_graph_nodes
from mjregrasping.movie import MjRenderer
from mjregrasping.mujoco_object import MjObject
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.physics import Physics
from mjregrasping.rerun_visualizer import MjReRun, log_skeletons
from mjregrasping.robot_data import drones


def main():
    rospy.init_node("drone_example")

    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    rr.init('drone_example')
    rr.connect()

    xml_path = 'models/drone_scene.xml'
    mjrr = MjReRun(xml_path)

    m = mujoco.MjModel.from_xml_path(xml_path)

    r = MjRenderer(m)

    m = mujoco.MjModel.from_xml_path(xml_path)

    objects = MjObjects(m, 'obstacles', drones, "rope")
    phy = Physics(m, mujoco.MjData(m), objects)
    mujoco.mj_forward(phy.m, phy.d)
    skeletons = get_tree_skeletons(phy)
    graph, h, loops = get_h_for_drones(phy, skeletons)
    viz_h(mjrr, phy, skeletons, graph, h, loops)

    # Release and move away from the pipe
    m.eq('drone1').active = False
    m.body("drone1").pos[0] = -3

    objects = MjObjects(m, 'obstacles', drones, "rope")
    phy = Physics(m, mujoco.MjData(m), objects)
    mujoco.mj_forward(phy.m, phy.d)
    skeletons = get_tree_skeletons(phy)
    graph, h, loops = get_h_for_drones(phy, skeletons)
    viz_h(mjrr, phy, skeletons, graph, h, loops)

    # Grasp the end, on the other side of the tree
    m.eq('drone1').active = True
    m.body("drone1").pos[0] = 0
    m.body("drone1").pos[1] = -14.4
    m.eq("drone1").obj2id = phy.m.body("B_1").id

    objects = MjObjects(m, 'obstacles', drones, "rope")
    phy = Physics(m, mujoco.MjData(m), objects)
    mujoco.mj_forward(phy.m, phy.d)
    skeletons = get_tree_skeletons(phy)
    graph, h, loops = get_h_for_drones(phy, skeletons)
    viz_h(mjrr, phy, skeletons, graph, h, loops)


def get_tree_skeletons(phy):
    skeletons = {
        'loop1': np.array([
            phy.d.site("branch0").xpos,
            phy.d.site("branch1").xpos,
            phy.d.site("branch2").xpos,
            phy.d.site("branch3").xpos,
            phy.d.site("branch4").xpos,
            phy.d.site("branch5").xpos,
            phy.d.site("branch0").xpos,
        ])
    }
    return skeletons


def viz_h(mjrr, phy, skeletons, graph, h, loops):
    print(h)

    mjrr.viz(phy, False, detailed=True)
    log_skeletons(skeletons)

    for i, loop_i in enumerate(loops):
        rr.log_line_strip(f'loop/{i}', loop_i, stroke_width=0.1)

    plt.figure()
    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.show()


def get_h_for_drones(phy, skeletons):
    graph = create_graph_nodes(phy)
    rope_points = get_rope_points(phy)
    # points tracing the robot arms from tip to base [n_arms, n_points, 3]
    base_xpos = phy.d.xpos[phy.m.body("drones").id][None]
    drone1_rope = MjObject(phy.m, 'drone1_rope')
    drone1_points = np.concatenate((phy.d.xpos[drone1_rope.body_indices[::-1]], base_xpos), 0)
    drone2_rope = MjObject(phy.m, 'drone2_rope')
    drone2_points = np.concatenate((phy.d.xpos[drone2_rope.body_indices[::-1]], base_xpos), 0)
    drone3_rope = MjObject(phy.m, 'drone3_rope')
    drone3_points = np.concatenate((phy.d.xpos[drone3_rope.body_indices[::-1]], base_xpos), 0)
    arm_points = np.stack([drone1_points, drone2_points, drone3_points])
    h, loops = get_full_h_signature(skeletons, graph, rope_points, arm_points,
                                    collapse_empty_gripper_cycles=False,
                                    gripper_ids_in_h_signature=False,
                                    connect_via_floor=False)
    return graph, h, loops


if __name__ == "__main__":
    main()
