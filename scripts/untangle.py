#!/usr/bin/env python3
import multiprocessing
from concurrent.futures.thread import ThreadPoolExecutor

import mujoco
import numpy as np
import rerun as rr
from colorama import Fore

from arc_utilities import ros_init
from mjregrasping.goals import GraspLocsGoal, point_goal_from_geom
from mjregrasping.grasp_and_settle import release_and_settle, grasp_and_settle
from mjregrasping.grasping import get_grasp_locs
from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner, get_geodesic_dist
from mjregrasping.move_to_joint_config import execute_grasp_plan
from mjregrasping.params import hp
from mjregrasping.regrasping_mppi import do_grasp_dynamics, RegraspMPPI, mppi_viz
from mjregrasping.rollout import control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import val_untangle
from mjregrasping.trap_detection import TrapDetection
from mjregrasping.trials import load_trial
from mjregrasping.viz import make_viz
from moveit_msgs.msg import MoveItErrorCodes


@ros_init.with_ros("untangle")
def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('untangle')
    rr.connect()

    scenario = val_untangle

    gl_ctx = mujoco.GLContext(1280, 720)
    gl_ctx.make_current()

    viz = make_viz(scenario)
    for i in range(10):
        # i = 999  # FIXME: debugging
        phy, sdf, skeletons, mov = load_trial(i, gl_ctx, scenario, viz)

        grasp_goal = GraspLocsGoal(get_grasp_locs(phy))
        goal = point_goal_from_geom(grasp_goal, phy, "goal", 1, viz)

        pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
        traps = TrapDetection()
        mppi = RegraspMPPI(pool=pool, nu=phy.m.nu, seed=i + 1, horizon=hp['horizon'],
                           noise_sigma=val_untangle.noise_sigma,
                           temp=hp['temp'])
        num_samples = hp['n_samples']
        grasp_rrt = GraspRRT()

        # planner = BaselineRegraspPlanner(goal, grasp_rrt, skeletons)
        planner = HomotopyRegraspPlanner(goal, grasp_rrt, skeletons)

        goal.viz_goal(phy)

        mppi.reset()
        traps.reset_trap_detection()

        itr = 0
        command = None
        sub_time_s = None
        success = False
        viz.viz(phy)
        while True:
            if itr >= scenario.max_iters:
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
                initial_geodesic_cost = get_geodesic_dist(grasp_goal.get_grasp_locs(), goal)
                sim_grasps = planner.simulate_sampled_grasps(phy, viz, viz_execution=True)
                best_grasp = planner.get_best(sim_grasps, viz=viz)
                new_geodesic_cost = get_geodesic_dist(best_grasp.locs, goal)
                # if we are unable to improve by grasping closer to the keypoint, update the blacklist and replan
                if initial_geodesic_cost - new_geodesic_cost < 0.01:  # less than 1% closer to the keypoint
                    print(Fore.YELLOW + "Unable to improve by grasping closer to the keypoint." + Fore.RESET)
                    print(Fore.YELLOW + "Updating blacklist and replanning..." + Fore.RESET)
                    planner.update_blacklists(phy)
                    best_grasp = planner.get_best(sim_grasps, viz=viz)
                if best_grasp.res.error_code.val != MoveItErrorCodes.SUCCESS:
                    print(Fore.RED + "Failed to find a plan." + Fore.RESET)
                viz.viz(best_grasp.phy, is_planning=True)
                # now execute the plan
                # from mjregrasping.trials import save_trial
                # save_trial(999, phy, scenario, None, skeletons)
                release_and_settle(phy, best_grasp.strategy, viz, is_planning=False, mov=mov)
                qs = np.array([p.positions for p in best_grasp.res.trajectory.joint_trajectory.points])
                execute_grasp_plan(phy, qs, viz, is_planning=False, mov=mov)
                grasp_and_settle(phy, best_grasp.locs, viz, is_planning=False, mov=mov)
                grasp_goal.set_grasp_locs(best_grasp.locs)

                needs_reset = True

            n_warmstart = max(1, min(hp['warmstart'], int((1 - traps.frac_dq) * 5)))

            if needs_reset:
                mppi.reset()
                traps.reset_trap_detection()
                n_warmstart = hp['warmstart']

            for k in range(n_warmstart):
                command, sub_time_s = mppi.command(phy, goal, num_samples, viz=viz)
                mppi_viz(mppi, goal, phy, command, sub_time_s)

            control_step(phy, command, sub_time_s, mov=mov)
            viz.viz(phy)

            results = goal.get_results(phy)
            do_grasp_dynamics(phy, results)

            mppi.roll()

            itr += 1

        metrics = {
            'itr':     itr,
            'success': success,
            'time':    phy.d.time
        }
        mov.close(metrics)


if __name__ == "__main__":
    main()
