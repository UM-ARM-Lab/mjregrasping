#!/usr/bin/env python3
import multiprocessing
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import rerun as rr
import rospy
import trimesh
from colorama import Fore
from mujoco import mj_id2name

from arc_utilities import ros_init
from geometry_msgs.msg import PointStamped, Point
from mjregrasping.goals import GraspLocsGoal, point_goal_from_geom, ThreadingGoal, ObjectPointGoal, get_disc_params
from mjregrasping.grasp_and_settle import grasp_and_settle, deactivate_release, deactivate_moving
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_grasp_locs
from mjregrasping.homotopy_checker import through_skels
from mjregrasping.homotopy_regrasp_planner import HomotopyThreadingPlanner
from mjregrasping.low_level_grasping import run_grasp_controller, GraspFailed
from mjregrasping.mjcf_scene_to_sdf import get_sdf, viz_slices
from mjregrasping.move_to_joint_config import pid_to_joint_configs
from mjregrasping.moveit_planning import make_planning_scene
from mjregrasping.movie import MjMovieMaker
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.params import hp
from mjregrasping.physics import Physics, get_q
from mjregrasping.real_val import RealValCommander
from mjregrasping.regrasping_mppi import RegraspMPPI, mppi_viz
from mjregrasping.rollout import control_step, get_speed_factor, DEFAULT_SUB_TIME_S
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import real_goal_sig, dz
from mjregrasping.set_up_real_scene import set_up_real_scene
from mjregrasping.trap_detection import TrapDetection
from mjregrasping.viz import make_viz
from moveit_msgs.msg import MoveItErrorCodes
from trimesh.sample import sample_surface


def get_real_untangle_skeletons(phy: Physics):
    d = phy.d
    m = phy.m
    return {
        "loop1": np.array([
            d.geom("loop1_front").xpos + dz(m.geom("loop1_front").size[2]),
            d.geom("loop1_front").xpos - dz(m.geom("loop1_front").size[2]),
            d.geom("loop1_back").xpos - dz(m.geom("loop1_back").size[2]),
            d.geom("loop1_back").xpos + dz(m.geom("loop1_back").size[2]),
            d.geom("loop1_front").xpos + dz(m.geom("loop1_front").size[2]),
        ]) - np.array([0.02, 0, 0]),  # hack to move the rope a bit further through the loop before the hand-off
    }


@ros_init.with_ros("untangle")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('untangle')
    rr.connect()

    scenario = real_goal_sig

    gl_ctx = mujoco.GLContext(1280, 720)
    gl_ctx.make_current()

    grasp_rrt = GraspRRT()

    hp['grasp_finger_weight'] = 0  # we aren't letting MPPI control the grippers in the real world
    hp['finger_q_open'] = 0.7
    hp['finger_q_closed'] = -0.05
    hp["threading_n_samples"] = 2
    hp['horizon'] = 7
    hp['n_samples'] = 32
    end_loc = 0.97

    viz = make_viz(scenario)
    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
    d = mujoco.MjData(m)
    phy = Physics(m, d, MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))
    mujoco.mj_forward(m, d)
    viz.viz(phy, True)
    val_cmd = RealValCommander(phy, pos_vmax=0.2)

    # from arc_utilities.tf2wrapper import TF2Wrapper
    # import rospy
    # tfw = TF2Wrapper()
    # rospy.sleep(1)
    #
    # ee_geom_names = [
    #     ['left_d405_mount', 'drive47_housing_mesh', 'drive47_mesh'],
    #     ['right_d405_mount', 'drive7_housing_mesh', 'drive7_mesh'],
    # ]
    # ee_link_names = [
    #     ['left_d405_mount', 'drive47', 'drive47'],
    #     ['right_d405_mount', 'drive7', 'drive7'],
    # ]
    # for geom_name, link_name in zip(ee_geom_names[0], ee_link_names[0]):
    #     mgeom = phy.m.geom(geom_name)
    #     mesh_name = mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mgeom.dataid)
    #     if mesh_name is None:
    #         continue
    #     if '/' in mesh_name:
    #         mesh_name = mesh_name.split("/")[1]
    #     mesh_file, mesh_scale = viz.mjrr.mj_xml_parser.get_mesh(mesh_name)
    #     mesh_file = Path(mesh_file).stem + ".stl"
    #     mesh_file = Path("models/meshes") / mesh_file
    #     with open(mesh_file, 'rb') as f:
    #         mesh = trimesh.load_mesh(f, file_type='stl')
    #     surface_points = sample_surface(mesh, 100)[0] * mesh_scale
    #
    #     # transform the surface points into the world frame
    #     p_in_link = PointStamped()
    #     p_in_link.header.frame_id = link_name
    #     surface_points_in_world = []
    #     for p in surface_points:
    #         p_in_link.point.x = float(p[0])
    #         p_in_link.point.y = float(p[1])
    #         p_in_link.point.z = float(p[2])
    #         p_in_world = tfw.transform_to_frame(p_in_link, "world")
    #         surface_points_in_world.append([p_in_world.point.x, p_in_world.point.y, p_in_world.point.z])
    #     surface_points_in_world = np.array(surface_points_in_world)
    #     viz.points('test_surface_points', surface_points_in_world, color=(0.4, 0.4, 0.4, 1), radius=0.005)
    #
    # sdf = get_sdf(phy, 0.01, -0.2, 0.15, 0.25, 0.7, 0.2, 0.8)
    # viz_slices(sdf)

    # FOR DEBUGGING GRASPING:
    # pos_in_mj_order = val_cmd.get_latest_qpos_in_mj_order()
    # val_cmd.update_mujoco_qpos(phy)
    # viz.viz(phy, False)
    # pid_to_joint_config(phy, viz, val_dedup(pos_in_mj_order), DEFAULT_SUB_TIME_S)
    # success_i = run_grasp_controller(val_cmd, phy, tool_idx=0, viz=viz, finger_q_open=hp['finger_q_open'], finger_q_closed=hp['finger_q_closed'])
    # return

    mov = None # MjMovieMaker(gl_ctx, phy.m) # this slows things down too much
    loc0 = 0.86
    set_up_real_scene(val_cmd, phy, viz)

    skeletons = get_real_untangle_skeletons(phy)
    viz.skeletons(skeletons)

    grasp_goal = GraspLocsGoal(get_grasp_locs(phy))
    goal = point_goal_from_geom(grasp_goal, phy, "goal", 0.97, viz)

    pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
    traps = TrapDetection()
    mppi = RegraspMPPI(pool=pool, nu=phy.m.nu, seed=1, horizon=hp['horizon'], noise_sigma=scenario.noise_sigma,
                       temp=hp['temp'])
    num_samples = hp['n_samples']

    goal.viz_goal(phy)

    mppi.reset()
    traps.reset_trap_detection()

    # DEBUGGING IK and Releasing
    # from mjregrasping.grasping import activate_grasp
    # val_cmd.pull_rope_towards_cdcpd(phy, 1000, settle_after=False)
    # viz.viz(phy)
    # activate_grasp(phy, 'right', 0.82)
    # val_cmd.pull_rope_towards_cdcpd(phy, 1000)
    # viz.viz(phy)
    # for _ in range(50):
    #     mujoco.mj_step(phy.m, phy.d, 20)
    #     viz.viz(phy)
    # val_cmd.set_cdcpd_from_mj_rope(phy)
    # phy_plan = phy.copy_all()
    # from mjregrasping.grasp_strategies import Strategies
    # from mjregrasping.grasp_and_settle import deactivate_release_and_moving
    # grasp_rrt.fix_start_state_in_place(phy_plan, viz)
    # strategy = [Strategies.NEW_GRASP, Strategies.STAY]
    # locs = np.array([end_loc, loc0])
    # res, scene_msg = grasp_rrt.plan(phy_plan, strategy, locs, viz, max_ik_attempts=1000, joint_noise=0.01,
    #                                 allowed_planning_time=30, pos_noise=0.01)
    # print(res.error_code.val)
    # grasp_rrt.display_result(viz, res, scene_msg)
    # deactivate_release_and_moving(phy, strategy, viz=viz, is_planning=False, val_cmd=val_cmd)
    # pid_to_joint_configs(phy, res, viz, is_planning=False, mov=mov, val_cmd=val_cmd)
    # run_grasp_controller(val_cmd, phy, tool_idx=0, viz=viz, finger_q_open=hp['finger_q_open'],
    #                      finger_q_closed=hp['finger_q_closed'], grasp_pos_inset=0.015)
    # return
    # grasp_and_settle(phy, locs, viz, is_planning=False, mov=mov, val_cmd=val_cmd)

    warn_near_joint_limits(phy)

    val_cmd.start_record(mov)

    goals = [
        ThreadingGoal(grasp_goal, skeletons, ['loop1'], end_loc, viz),
        point_goal_from_geom(grasp_goal, phy, "goal", end_loc, viz)
    ]

    traps = TrapDetection()

    itr = 0
    goal_idx = 0
    goal = goals[goal_idx]
    needs_reset = True

    viz.viz(phy)
    while True:
        if itr > 275:
            print(Fore.RED + "Max iterations reached!" + Fore.RESET)
            break

        goal.viz_goal(phy)

        if isinstance(goal, ObjectPointGoal):
            if goal.satisfied(phy):
                print(Fore.GREEN + "Task complete!" + Fore.RESET)
                break
        else:
            disc_center, disc_normal = get_disc_params(goal)
            disc_rad = 0.30

            disc_penetrated = goal.satisfied(phy, disc_center, disc_normal, disc_rad, end_loc)
            is_stuck = traps.check_is_stuck(phy, grasp_goal)
            if disc_penetrated:
                print("Disc penetrated!")
                mppi.reset()

                # open the left gripper
                qvel_open = np.zeros(phy.m.nu)
                qvel_open[9] = 0.5
                control_step(phy, qvel_open, sub_time_s=2, mov=None, val_cmd=val_cmd)

                planner = HomotopyThreadingPlanner(end_loc, grasp_rrt, skeletons, goal.skeleton_names)
                print(f"Planning with {planner.key_loc=}...")
                sim_grasps = planner.simulate_sampled_grasps(phy, viz, viz_execution=True)
                best_grasp = planner.get_best(sim_grasps, viz=viz)
                viz.viz(best_grasp.phy, is_planning=True)
                # now execute the plan
                if best_grasp.res.error_code.val == MoveItErrorCodes.SUCCESS:
                    scene_msg = make_planning_scene(phy)
                    grasp_rrt.display_result(viz, best_grasp.res, scene_msg)
                    print(f"Regrasping from {best_grasp.initial_locs} to {best_grasp.locs}")
                    if planner.through_skels(best_grasp.phy):
                        print("Executing grasp change plan")
                        execute_grasp_change_plan(best_grasp, grasp_goal, phy, viz, mov, val_cmd, grasp_pos_inset=0.02)
                        needs_reset = True
                        traps.reset_trap_detection()
                    else:
                        print("Not through the goal skeleton!")
                else:
                    # if we've reached the goal but can't grasp the end, scootch down the rope
                    # but don't scootch past where we are currently grasping it.
                    goal.loc = max(goal.loc - hp['ours_scootch_fraction'], max(get_grasp_locs(phy)))
                    print("No plans found!")
            elif np.all(grasp_goal.get_grasp_locs() == -1):
                print("Doing initial grasp!")
                mppi.reset()

                # open the right gripper
                qvel_open = np.zeros(phy.m.nu)
                qvel_open[-1] = 0.5
                control_step(phy, qvel_open, sub_time_s=2, mov=None, val_cmd=val_cmd)

                planner = HomotopyThreadingPlanner(loc0, grasp_rrt, skeletons, goal.skeleton_names)
                print(f"Planning with {planner.key_loc=}...")
                sim_grasps = planner.simulate_sampled_grasps(phy, viz, viz_execution=False)
                best_grasp = planner.get_best(sim_grasps, viz=viz)
                viz.viz(best_grasp.phy, is_planning=True)
                # now execute the plan
                if best_grasp.res.error_code.val == MoveItErrorCodes.SUCCESS:
                    scene_msg = make_planning_scene(phy)
                    grasp_rrt.display_result(viz, best_grasp.res, scene_msg)
                    print(f"Regrasping from {best_grasp.initial_locs} to {best_grasp.locs}")
                    # temporarily disable collision for the grippers
                    phy.m.geom('rightgripper').contype = 0
                    phy.m.geom('rightgripper2').contype = 0
                    execute_grasp_change_plan(best_grasp, grasp_goal, phy, viz, mov, val_cmd, grasp_pos_inset=0.014)
                    phy.m.geom('rightgripper').contype = 1  # re-enable
                    phy.m.geom('rightgripper2').contype = 1
                    traps.reset_trap_detection()
                    needs_reset = True
                else:
                    raise RuntimeError("No plans found!")

            if through_skels(skeletons, goal.skeleton_names, phy):
                print(f"Through {goal.skeleton_names}!")
                goal_idx += 1
                goal = goals[goal_idx]

        if needs_reset:
            for _ in range(6):
                command, sub_time_s = mppi.command(phy, goal, num_samples, viz=viz)
                mppi_viz(mppi, goal, phy, command, sub_time_s)
            needs_reset = False
        command, sub_time_s = mppi.command(phy, goal, num_samples, viz=viz)
        mppi_viz(mppi, goal, phy, command, sub_time_s)

        control_step(phy, command, sub_time_s, mov=mov, val_cmd=val_cmd, slow=True)
        viz.viz(phy)

        mppi.roll()

        val_cmd.set_cdcpd_from_mj_rope(phy)

        speed_factor = get_speed_factor(phy)
        rr.log_scalar('speed_factor', speed_factor)

        itr += 1

    val_cmd.disconnect()

    rospy.sleep(5)

    val_cmd.stop_record()


def execute_grasp_change_plan(best_grasp, grasp_goal, phy, viz, mov, val_cmd, grasp_pos_inset: float):
    deactivate_moving(phy, best_grasp.strategy, viz, is_planning=False, mov=mov, val_cmd=val_cmd)
    pid_to_joint_configs(phy, best_grasp.res, viz, is_planning=False, mov=mov, val_cmd=val_cmd)
    for i, s_i in enumerate(best_grasp.strategy):
        if s_i in [Strategies.MOVE, Strategies.NEW_GRASP]:
            while True:
                try:
                    success_i = run_grasp_controller(val_cmd, phy, tool_idx=i, viz=viz,
                                                     finger_q_open=hp['finger_q_open'],
                                                     finger_q_closed=hp['finger_q_closed'],
                                                     grasp_pos_inset=grasp_pos_inset)
                    if not success_i:
                        raise GraspFailed("Grasp controller failed!")
                    break
                except RuntimeError:
                    print("Realsense issue? Try unplugging and replugging the camera.")
    grasp_and_settle(phy, best_grasp.locs, viz, is_planning=False, mov=mov, val_cmd=val_cmd)
    deactivate_release(phy, best_grasp.strategy, viz=viz, is_planning=False, mov=mov, val_cmd=val_cmd)
    grasp_goal.set_grasp_locs(best_grasp.locs)
    print(f"Changed grasp to {best_grasp.locs}")


def warn_near_joint_limits(phy):
    current_q = get_q(phy)
    near_low = current_q < -4
    near_high = current_q > 4
    if np.any(near_low):
        i = np.where(near_low)[0][0]
        name_i = phy.m.actuator(i).name
        q_i = current_q[i]
        limit_i = phy.m.actuator_actrange[i, 0]
        if 'gripper' not in name_i:
            print(f"WARNING: joint {name_i} is at {q_i}, near lower limit of {limit_i}!")
    elif np.any(near_high):
        i = np.where(near_high)[0][0]
        name_i = phy.m.actuator(i).name

        q_i = current_q[i]
        limit_i = phy.m.actuator_actrange[i, 1]
        if 'gripper' not in name_i:
            print(f"WARNING: joint {name_i} is at {q_i}, near upper limit of {limit_i}!")


if __name__ == "__main__":
    main()
