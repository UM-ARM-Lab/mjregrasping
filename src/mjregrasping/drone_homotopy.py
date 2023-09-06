import numpy as np

from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.homotopy_checker import get_full_h_signature, create_graph_nodes
from mjregrasping.mujoco_object import MjObject


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


def get_h_for_drones(phy, skeletons):
    from time import perf_counter
    t0 = perf_counter()
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
    print(f'dt: {perf_counter() - t0:.3f}')
    return graph, h, loops
