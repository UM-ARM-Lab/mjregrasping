#!/usr/bin/env python3
import multiprocessing
from concurrent.futures.thread import ThreadPoolExecutor
from time import perf_counter, sleep
from typing import Optional, List

import mujoco
import numpy as np
import rerun as rr
from colorama import Fore

# noinspection PyUnresolvedReferences
import tf2_geometry_msgs
from arc_utilities import ros_init
from mjregrasping.goals import GraspLocsGoal, point_goal_from_geom
from mjregrasping.grasp_and_settle import grasp_and_settle, deactivate_release_and_moving
from mjregrasping.grasp_strategies import Strategies
from mjregrasping.grasping import get_grasp_locs, activate_grasp
from mjregrasping.low_level_grasping import run_grasp_controller
from mjregrasping.move_to_joint_config import pid_to_joint_configs, pid_to_joint_config
from mjregrasping.movie import MjMovieMaker
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.real_val import RealValCommander
from mjregrasping.regrasp_planner_utils import get_geodesic_dist
from mjregrasping.regrasping_mppi import do_grasp_dynamics, RegraspMPPI, mppi_viz
from mjregrasping.rollout import control_step, get_speed_factor, DEFAULT_SUB_TIME_S
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import real_goal_sig, dz
from mjregrasping.set_up_real_scene import set_up_real_scene
from mjregrasping.trap_detection import TrapDetection
from mjregrasping.val_dup import val_dedup
from mjregrasping.viz import make_viz, Viz
from moveit_msgs.msg import MoveItErrorCodes


class BaseOnStuckMethod:

    def __init__(self, scenario, skeletons, goal, grasp_goal, grasp_rrt: GraspRRT):
        self.scenario = scenario
        self.skeletons = skeletons
        self.goal = goal
        self.grasp_goal = grasp_goal
        self.grasp_rrt = grasp_rrt

    def on_stuck(self, phy: Physics, viz: Viz, mov: Optional[MjMovieMaker], val_cmd: Optional[RealValCommander]):
        raise NotImplementedError()


class OnStuckOurs(BaseOnStuckMethod):

    def __init__(self, scenario, skeletons, goal, grasp_goal, grasp_rrt: GraspRRT,
                 goal_skel_names: Optional[List[str]] = None, **ik_kwargs):
        super().__init__(scenario, skeletons, goal, grasp_goal, grasp_rrt)
        from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner
        self.planner = HomotopyRegraspPlanner(goal.loc, self.grasp_rrt, skeletons, goal_skel_names, **ik_kwargs)

    def on_stuck(self, phy, viz, mov, val_cmd: Optional[RealValCommander]):
        initial_geodesic_dist = get_geodesic_dist(self.grasp_goal.get_grasp_locs(), self.goal.loc)
        planning_t0 = perf_counter()
        sim_grasps = self.planner.simulate_sampled_grasps(phy, viz, viz_execution=True)
        best_grasp = self.planner.get_best(sim_grasps, viz=viz)
        new_geodesic_dist = get_geodesic_dist(best_grasp.locs, self.goal.loc)
        # if we are unable to improve by grasping closer to the keypoint, update the blacklist and replan
        if initial_geodesic_dist - new_geodesic_dist < hp['grasp_loc_diff_thresh']:
            print(Fore.YELLOW + "Unable to improve by grasping closer to the keypoint." + Fore.RESET)
            print(Fore.YELLOW + "Updating blacklist and re-evaluating cost..." + Fore.RESET)
            self.planner.update_blacklists(phy)
            best_grasp = self.planner.get_best(sim_grasps, viz=viz)
        self.planner.planning_times.append(perf_counter() - planning_t0)
        if best_grasp.res.error_code.val == MoveItErrorCodes.SUCCESS:
            viz.viz(best_grasp.phy, is_planning=True)
            from mjregrasping.moveit_planning import make_planning_scene
            scene_msg = make_planning_scene(phy)
            self.grasp_rrt.display_result(viz, best_grasp.res, scene_msg)
            print(f"Regrasping from {best_grasp.initial_locs} to {best_grasp.locs}")
            # now execute the plan
            deactivate_release_and_moving(phy, best_grasp.strategy, viz, is_planning=False, mov=mov, val_cmd=val_cmd)
            pid_to_joint_configs(phy, best_grasp.res, viz, is_planning=False, mov=mov, val_cmd=val_cmd)
            ##################################################
            # Run a low level controller to actually grasp
            for i, s_i in enumerate(best_grasp.strategy):
                if s_i in [Strategies.MOVE, Strategies.NEW_GRASP]:
                    while True:
                        try:
                            success_i = run_grasp_controller(val_cmd, phy, tool_idx=i, viz=viz, finger_q_open=0.5,
                                                             finger_q_closed=0.0)
                            if not success_i:
                                print("Grasp controller failed!")
                                return
                            break
                        except RuntimeError:
                            print("Realsense issue? Try unplugging and replugging the camera.")
            ##################################################
            grasp_and_settle(phy, best_grasp.locs, viz, is_planning=False, mov=mov,
                             val_cmd=None)  # we don't need val to grasp, that's what the grasp controller is for
            val_cmd.set_cdcpd_grippers(phy)
            self.grasp_goal.set_grasp_locs(best_grasp.locs)
        else:
            print(Fore.RED + "Failed to find a plan." + Fore.RESET)


def get_real_untangle_skeletons(phy: Physics):
    d = phy.d
    m = phy.m
    return {
        "loop1": np.array([
            d.geom("loop1_front").xpos - dz(m.geom("loop1_front").size[2]),
            d.geom("loop1_front").xpos + dz(m.geom("loop1_front").size[2]),
            d.geom("loop1_back").xpos + dz(m.geom("loop1_back").size[2]),
            d.geom("loop1_back").xpos - dz(m.geom("loop1_back").size[2]),
            d.geom("loop1_front").xpos - dz(m.geom("loop1_front").size[2]),
        ]),
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

    viz = make_viz(scenario)
    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
    d = mujoco.MjData(m)
    phy = Physics(m, d, MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))
    val_cmd = RealValCommander(phy)

    mov = None
    loc0 = 0.8
    set_up_real_scene(val_cmd, phy, viz, loc0)

    skeletons = get_real_untangle_skeletons(phy)
    viz.skeletons(skeletons)

    val_cmd.set_cdcpd_grippers(phy)
    val_cmd.set_cdcpd_from_mj_rope(phy)
    # let CDCPD converge
    sleep(5)
    val_cmd.pull_rope_towards_cdcpd(phy, 800)
    viz.viz(phy)

    grasp_goal = GraspLocsGoal(get_grasp_locs(phy))
    goal = point_goal_from_geom(grasp_goal, phy, "goal", 0.95, viz)

    pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
    traps = TrapDetection()
    hp['finger_q_pregrasp'] = 0.3
    hp['grasp_finger_weight'] = 0  # we aren't letting MPPI control the grippers in the real world
    hp['finger_q_open'] = 0.5
    hp['finger_q_closed'] = 0
    hp['horizon'] = 5
    hp['n_samples'] = 32
    hp['n_grasp_samples'] = 1
    mppi = RegraspMPPI(pool=pool, nu=phy.m.nu, seed=1, horizon=hp['horizon'], noise_sigma=scenario.noise_sigma,
                       temp=hp['temp'])
    num_samples = hp['n_samples']
    osm = OnStuckOurs(scenario, skeletons, goal, grasp_goal, grasp_rrt, goal_skel_names=[], max_ik_attempts=1500,
                      joint_noise=0.01, allowed_planning_time=15, pos_noise=0.04)

    goal.viz_goal(phy)

    mppi.reset()
    traps.reset_trap_detection()

    # DEBUGGING IK and Releasing
    phy_plan = phy.copy_all()
    from mjregrasping.grasp_strategies import Strategies
    # grasp_rrt.fix_start_state_in_place(phy_plan, viz)
    strategy = [Strategies.NEW_GRASP, Strategies.STAY]
    locs = np.array([1, loc0])
    res, scene_msg = grasp_rrt.plan(phy_plan, strategy, locs, viz, max_ik_attempts=1000, joint_noise=0.01, allowed_planning_time=30)
    print(res.error_code.val)
    grasp_rrt.display_result(viz, res, scene_msg)
    deactivate_release_and_moving(phy, strategy, viz=viz, is_planning=False, val_cmd=val_cmd)
    pid_to_joint_configs(phy, res, viz, is_planning=False, mov=mov, val_cmd=val_cmd)
    run_grasp_controller(val_cmd, phy, tool_idx=1, viz=viz, finger_q_open=hp['finger_q_open'], finger_q_closed=hp['finger_q_closed'])
    grasp_and_settle(phy, locs, viz, is_planning=False, mov=mov, val_cmd=val_cmd)
    return

    val_cmd.start_record()

    itr = 0
    viz.viz(phy)
    while True:
        if itr >= 300:
            print(Fore.RED + "Task failed!" + Fore.RESET)
            break

        goal.viz_goal(phy)
        if goal.satisfied(phy):
            print(Fore.GREEN + "Task Complete!" + Fore.RESET)
            break

        is_stuck = traps.check_is_stuck(phy, grasp_goal)
        needs_reset = False
        if is_stuck:
            print(Fore.YELLOW + "Stuck! Replanning..." + Fore.RESET)
            osm.on_stuck(phy, viz, mov, val_cmd)
            needs_reset = True

        if needs_reset:
            mppi.reset()
            traps.reset_trap_detection()

        mpc_t0 = perf_counter()
        if itr == 0:  # improves the first few commands. not needed, but nice for real world demo
            for _ in range(5):
                command, sub_time_s = mppi.command(phy, goal, num_samples, viz=viz)
                mppi_viz(mppi, goal, phy, command, sub_time_s)
        command, sub_time_s = mppi.command(phy, goal, num_samples, viz=viz)
        mppi_viz(mppi, goal, phy, command, sub_time_s)
        mpc_dt = perf_counter() - mpc_t0
        print(f'mppi.command {mpc_dt:.3f}s')

        control_step(phy, command, sub_time_s, mov=mov, val_cmd=val_cmd)
        viz.viz(phy)

        speed_factor = get_speed_factor(phy)
        rr.log_scalar('speed_factor', speed_factor)

        do_grasp_dynamics(phy, val_cmd)

        mppi.roll()

        itr += 1

    val_cmd.stop_record()


if __name__ == "__main__":
    main()
