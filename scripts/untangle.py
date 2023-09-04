#!/usr/bin/env python3
import multiprocessing
from concurrent.futures.thread import ThreadPoolExecutor
from time import perf_counter
from typing import Optional

import mujoco
import numpy as np
import rerun as rr
from colorama import Fore

import rospy
from arc_utilities import ros_init
from arc_utilities.listener import Listener
from mjregrasping.goals import GraspLocsGoal, point_goal_from_geom
from mjregrasping.grasp_and_settle import grasp_and_settle, deactivate_release_and_moving
from mjregrasping.grasping import get_grasp_locs, activate_grasp
from mjregrasping.move_to_joint_config import execute_grasp_plan
from mjregrasping.movie import MjMovieMaker
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.real_val import RealValCommander, update_mujoco_qpos
from mjregrasping.regrasp_planner_utils import get_geodesic_dist
from mjregrasping.regrasping_mppi import do_grasp_dynamics, RegraspMPPI, mppi_viz
from mjregrasping.rollout import control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import val_untangle, real_untangle, get_real_untangle_skeletons
from mjregrasping.trap_detection import TrapDetection
from mjregrasping.trials import load_trial
from mjregrasping.viz import make_viz, Viz
from moveit_msgs.msg import MoveItErrorCodes
from ros_numpy import numpify
from visualization_msgs.msg import MarkerArray


class BaseOnStuckMethod:

    def __init__(self, scenario, skeletons, goal, grasp_goal, grasp_rrt: GraspRRT):
        self.scenario = scenario
        self.skeletons = skeletons
        self.goal = goal
        self.grasp_goal = grasp_goal
        self.grasp_rrt = grasp_rrt

    def on_stuck(self, phy: Physics, viz: Viz, mov: Optional[MjMovieMaker]):
        raise NotImplementedError()


class OnStuckTamp(BaseOnStuckMethod):

    def __init__(self, scenario, skeletons, goal, grasp_goal, grasp_rrt: GraspRRT):
        super().__init__(scenario, skeletons, goal, grasp_goal, grasp_rrt)
        from mjregrasping.tamp_regrasp_planner import TAMPRegraspPlanner
        self.planner = TAMPRegraspPlanner(scenario, goal, self.grasp_rrt, skeletons)

    def on_stuck(self, phy, viz, mov):
        planning_t0 = perf_counter()
        sim_grasps = self.planner.simulate_sampled_grasps(phy, viz, viz_execution=True)
        best_grasp = self.planner.get_best(sim_grasps, viz=viz)
        self.planner.planning_times.append(perf_counter() - planning_t0)

        if best_grasp.res.error_code.val == MoveItErrorCodes.SUCCESS:
            viz.viz(best_grasp.phy, is_planning=True)
            deactivate_release_and_moving(phy, best_grasp.strategy, viz, is_planning=False, mov=mov)
            execute_grasp_plan(phy, best_grasp.res, viz, is_planning=False, mov=mov)
            grasp_and_settle(phy, best_grasp.locs, viz, is_planning=False, mov=mov)
            self.grasp_goal.set_grasp_locs(best_grasp.locs)
        else:
            print(Fore.RED + "Failed to find a plan." + Fore.RESET)


class OnStuckOurs(BaseOnStuckMethod):

    def __init__(self, scenario, skeletons, goal, grasp_goal, grasp_rrt: GraspRRT):
        super().__init__(scenario, skeletons, goal, grasp_goal, grasp_rrt)
        from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner
        self.planner = HomotopyRegraspPlanner(goal.loc, self.grasp_rrt, skeletons)

    def on_stuck(self, phy, viz, mov):
        initial_geodesic_dist = get_geodesic_dist(self.grasp_goal.get_grasp_locs(), self.goal.loc)
        planning_t0 = perf_counter()
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
            # now execute the plan
            deactivate_release_and_moving(phy, best_grasp.strategy, viz, is_planning=False, mov=mov)
            execute_grasp_plan(phy, best_grasp.res, viz, is_planning=False, mov=mov)
            grasp_and_settle(phy, best_grasp.locs, viz, is_planning=False, mov=mov)
            self.grasp_goal.set_grasp_locs(best_grasp.locs)
        else:
            print(Fore.RED + "Failed to find a plan." + Fore.RESET)


def set_mujoco_rope_state_from_cdcpd(cdcpd_pred: MarkerArray, phy: Physics, viz: Viz):
    cdcpd_np = []
    for marker in cdcpd_pred.markers:
        cdcpd_np.append(numpify(marker.pose.position))
    cdcpd_np = np.array(cdcpd_np)




@ros_init.with_ros("untangle")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('untangle')
    rr.connect()

    # scenario = val_untangle
    scenario = real_untangle

    gl_ctx = mujoco.GLContext(1280, 720)
    gl_ctx.make_current()

    grasp_rrt = GraspRRT()

    viz = make_viz(scenario)
    for trial_idx in range(0, 10):
        if scenario == val_untangle:
            phy, _, skeletons, mov = load_trial(trial_idx, gl_ctx, scenario, viz)
        else:
            val = RealValCommander(phy.o.robot)
            cdcpd_sub = Listener("/cdcpd_pred", MarkerArray)

            m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
            d = mujoco.MjData(m)
            phy = Physics(m, d, MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name))

            skeletons = get_real_untangle_skeletons(phy)
            mov = None
            set_up_real_scene(val, phy, viz)

        grasp_goal = GraspLocsGoal(get_grasp_locs(phy))
        goal = point_goal_from_geom(grasp_goal, phy, "goal", 1, viz)

        cdcpd_pred = cdcpd_sub.get()
        set_mujoco_rope_state_from_cdcpd(cdcpd_pred, phy, viz)

        pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
        traps = TrapDetection()
        mppi = RegraspMPPI(pool=pool, nu=phy.m.nu, seed=trial_idx + 1, horizon=hp['horizon'],
                           noise_sigma=val_untangle.noise_sigma,
                           temp=hp['temp'])
        num_samples = hp['n_samples']
        osm = OnStuckOurs(scenario, skeletons, goal, grasp_goal, grasp_rrt)
        # osm = OnStuckTamp(scenario, skeletons, goal, grasp_goal)
        mpc_times = []

        goal.viz_goal(phy)

        mppi.reset()
        traps.reset_trap_detection()

        itr = 0
        success = False
        viz.viz(phy)
        while True:
            if itr >= 200:
                print(Fore.RED + "Task failed!" + Fore.RESET)
                break

            goal.viz_goal(phy)
            if goal.satisfied(phy):
                success = True
                print(Fore.GREEN + "Task Complete!" + Fore.RESET)
                break

            is_stuck = traps.check_is_stuck(phy)
            needs_reset = False
            if is_stuck:
                print(Fore.YELLOW + "Stuck! Replanning..." + Fore.RESET)
                osm.on_stuck(phy, viz, mov)
                needs_reset = True

            if needs_reset:
                mppi.reset()
                traps.reset_trap_detection()

            mpc_t0 = perf_counter()
            command, sub_time_s = mppi.command(phy, goal, num_samples, viz=viz)
            mpc_times.append(perf_counter() - mpc_t0)
            mppi_viz(mppi, goal, phy, command, sub_time_s)

            control_step(phy, command, sub_time_s, mov=mov)
            viz.viz(phy)

            do_grasp_dynamics(phy)

            mppi.roll()

            itr += 1

        metrics = {
            'itr':            itr,
            'success':        success,
            'sim_time':       phy.d.time,
            'planning_times': osm.planner.planning_times,
            'mpc_times':      mpc_times,
            'grasp_history':  np.array(grasp_goal.history).tolist(),
            'method':        osm.__class__.__name__,
        }
        print(metrics)
        if mov:
            mov.close(metrics)


def set_up_real_scene(val: RealValCommander, phy: Physics, viz: Viz):
    update_mujoco_qpos(phy, val)
    viz.viz(phy, False)
    for _ in range(100):
        mujoco.mj_step(phy.m, phy.d, 5)
        viz.viz(phy, False)
    activate_grasp(phy, 'right', 1)
    for _ in range(100):
        mujoco.mj_step(phy.m, phy.d, 5)
        viz.viz(phy, False)


if __name__ == "__main__":
    main()
