#!/usr/bin/env python3
import multiprocessing
from concurrent.futures.thread import ThreadPoolExecutor
from time import perf_counter
from typing import Optional

import mujoco
import numpy as np
import rerun as rr
from colorama import Fore

# noinspection PyUnresolvedReferences
import tf2_geometry_msgs
from arc_utilities import ros_init
from mjregrasping.goals import GraspLocsGoal, point_goal_from_geom
from mjregrasping.grasp_and_settle import grasp_and_settle, deactivate_release_and_moving
from mjregrasping.grasping import get_grasp_locs
from mjregrasping.move_to_joint_config import pid_to_joint_configs
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.real_val import RealValCommander
from mjregrasping.regrasp_planner_utils import get_geodesic_dist
from mjregrasping.regrasping_mppi import do_grasp_dynamics, RegraspMPPI, mppi_viz
from mjregrasping.rollout import control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import val_untangle, real_untangle, get_real_untangle_skeletons
from mjregrasping.set_up_real_scene import set_up_real_scene
from mjregrasping.trap_detection import TrapDetection
from mjregrasping.untangle_methods import BaseOnStuckMethod
from mjregrasping.viz import make_viz
from moveit_msgs.msg import MoveItErrorCodes


class OnStuckReal(BaseOnStuckMethod):

    def __init__(self, scenario, skeletons, goal, grasp_goal, grasp_rrt: GraspRRT):
        super().__init__(scenario, skeletons, goal, grasp_goal, grasp_rrt)
        from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner
        self.planner = HomotopyRegraspPlanner(goal.loc, self.grasp_rrt, skeletons)

    def on_stuck(self, phy, viz, mov, val_cmd: Optional[RealValCommander] = None):
        initial_geodesic_dist = get_geodesic_dist(self.grasp_goal.get_grasp_locs(), self.goal.loc)
        planning_t0 = perf_counter()
        # print("DEBUGGING VIZ_EXECUTION=TRUE")
        sim_grasps = self.planner.simulate_sampled_grasps(phy, viz, viz_execution=False)
        best_grasp = self.planner.get_best(sim_grasps, viz=viz)
        new_geodesic_dist = get_geodesic_dist(best_grasp.locs, self.goal.loc)
        # if we are unable to improve by grasping closer to the keypoint, update the blacklist and replan
        if initial_geodesic_dist - new_geodesic_dist < 0.01:  # less than 1% closer to the keypoint
            print(Fore.YELLOW + "Unable to improve by grasping closer to the keypoint." + Fore.RESET)
            print(Fore.YELLOW + "Updating blacklist and replanning..." + Fore.RESET)
            self.planner.update_blacklists(phy)
            best_grasp = self.planner.get_best(sim_grasps, viz=viz)
        self.planner.planning_times.append(perf_counter() - planning_t0)
        if best_grasp.res.error_code.val == MoveItErrorCodes.SUCCESS:
            viz.viz(best_grasp.phy, is_planning=True)
            print(f"Regrasping from {best_grasp.initial_locs} to {best_grasp.locs}")
            # now execute the plan
            deactivate_release_and_moving(phy, best_grasp.strategy, viz, is_planning=False, mov=mov, val_cmd=val_cmd)
            pid_to_joint_configs(phy, best_grasp.res, viz, is_planning=False, mov=mov, val_cmd=val_cmd)
            grasp_and_settle(phy, best_grasp.locs, viz, is_planning=False, mov=mov, val_cmd=val_cmd)
            self.grasp_goal.set_grasp_locs(best_grasp.locs)
        else:
            print(Fore.RED + "Failed to find a plan." + Fore.RESET)


@ros_init.with_ros("untangle")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('untangle')
    rr.connect()

    scenario = real_untangle

    gl_ctx = mujoco.GLContext(1280, 720)
    gl_ctx.make_current()

    grasp_rrt = GraspRRT()

    viz = make_viz(scenario)
    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
    d = mujoco.MjData(m)
    phy = Physics(m, d, MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))
    val_cmd = RealValCommander(phy)

    mov = None
    set_up_real_scene(val_cmd, phy, viz)

    skeletons = get_real_untangle_skeletons(phy)
    viz.skeletons(skeletons)

    val_cmd.set_cdcpd_from_mj_rope(phy)

    grasp_goal = GraspLocsGoal(get_grasp_locs(phy))
    goal = point_goal_from_geom(grasp_goal, phy, "goal", 1, viz)

    pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
    traps = TrapDetection()
    hp['horizon'] = 8;
    hp['n_samples'] = 36
    mppi = RegraspMPPI(pool=pool, nu=phy.m.nu, seed=1, horizon=hp['horizon'], noise_sigma=val_untangle.noise_sigma,
                       temp=hp['temp'])
    num_samples = hp['n_samples']
    osm = OnStuckReal(scenario, skeletons, goal, grasp_goal, grasp_rrt)

    goal.viz_goal(phy)

    mppi.reset()
    traps.reset_trap_detection()

    itr = 0
    viz.viz(phy)
    while True:
        val_cmd.update_mujoco_qpos(phy)

        if itr >= 300:
            print(Fore.RED + "Task failed!" + Fore.RESET)
            break

        goal.viz_goal(phy)
        if goal.satisfied(phy):
            print(Fore.GREEN + "Task Complete!" + Fore.RESET)
            break

        is_stuck = traps.check_is_stuck(phy)
        needs_reset = False
        if itr == 10 or is_stuck:
            print("DEBUGGING!!!")
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

        do_grasp_dynamics(phy, val_cmd)

        mppi.roll()

        itr += 1


if __name__ == "__main__":
    main()
