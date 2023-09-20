#!/usr/bin/env python3
import argparse
import multiprocessing
from concurrent.futures.thread import ThreadPoolExecutor
from time import perf_counter

import mujoco
import numpy as np
import rerun as rr
from colorama import Fore

from arc_utilities import ros_init
from mjregrasping.goals import GraspLocsGoal, point_goal_from_geom
from mjregrasping.grasping import get_grasp_locs
from mjregrasping.params import hp
from mjregrasping.regrasping_mppi import do_grasp_dynamics, RegraspMPPI, mppi_viz
from mjregrasping.rollout import control_step
from mjregrasping.rrt import GraspRRT
from mjregrasping.scenarios import val_untangle, simple_goal_sig
from mjregrasping.trap_detection import TrapDetection
from mjregrasping.trials import load_trial
from mjregrasping.point_reaching_methods import OnStuckOurs, OnStuckTamp, OnStuckAlwaysBlacklist, OursNoSignature
from mjregrasping.viz import make_viz


@ros_init.with_ros("untangle")
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("scenario")
    parser.add_argument("method")

    args = parser.parse_args()

    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    rr.init('untangle')
    rr.connect()

    if args.scenario == "val_untangle":
        scenario = val_untangle
    elif args.scenario == "simple_goal_sig":
        scenario = simple_goal_sig
    else:
        raise NotImplementedError(f"Unknown scenario: {args.scenario}")

    gl_ctx = mujoco.GLContext(1280, 720)
    gl_ctx.make_current()

    grasp_rrt = GraspRRT()

    viz = make_viz(scenario)
    for trial_idx in [15, 18, 2, 11, 12, 17]:
    # for trial_idx in range(0, 25):
        phy, _, skeletons, mov = load_trial(trial_idx, gl_ctx, scenario, viz)
    
        # disable collision only between rope and gripper geoms
        from itertools import chain
        for geom_name in list(chain(*phy.o.rd.gripper_geom_names)) + phy.o.rope.geom_names:
            phy.m.geom(geom_name).contype = 1
            phy.m.geom(geom_name).conaffinity = 2

        overall_t0 = perf_counter()

        grasp_goal = GraspLocsGoal(get_grasp_locs(phy))
        goal = point_goal_from_geom(grasp_goal, phy, "goal", 1, viz)

        pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)
        traps = TrapDetection()
        mppi = RegraspMPPI(pool=pool, nu=phy.m.nu, seed=trial_idx + 1, horizon=hp['horizon'],
                           noise_sigma=val_untangle.noise_sigma,
                           temp=hp['temp'])
        num_samples = hp['n_samples']

        if args.method == "ours":
            osm = OnStuckOurs(scenario, skeletons, goal, grasp_goal, grasp_rrt)
        elif args.method == "tamp":
            osm = OnStuckTamp(scenario, skeletons, goal, grasp_goal, grasp_rrt)
        elif args.method == "blacklist":
            osm = OnStuckAlwaysBlacklist(scenario, skeletons, goal, grasp_goal, grasp_rrt)
        elif args.method == "no_signature":
            osm = OursNoSignature(scenario, skeletons, goal, grasp_goal, grasp_rrt)
        else:
            raise NotImplementedError(f"Unknown method: {args.method}")
        print(Fore.BLUE + f"Running method {osm.method_name()} with {scenario.name}" + Fore.RESET)
        mpc_times = []

        goal.viz_goal(phy)

        mppi.reset()
        traps.reset_trap_detection()

        itr = 0
        success = False
        viz.viz(phy)
        while True:
            if itr >= hp["untangle_max_iters"]:
                print(Fore.RED + "Task failed!" + Fore.RESET)
                break

            goal.viz_goal(phy)
            if goal.satisfied(phy):
                success = True
                print(Fore.GREEN + "Task Complete!" + Fore.RESET)
                break

            is_stuck = traps.check_is_stuck(phy, grasp_goal)
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
            mppi_viz(mppi, goal, phy, command, sub_time_s)
            mpc_times.append(perf_counter() - mpc_t0)

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
            'overall_time':   perf_counter() - overall_t0,
            'grasp_history':  np.array(grasp_goal.history).tolist(),
            'method':         osm.method_name(),
            'hp':             hp,
        }
        print(metrics)
        if mov:
            mov.close(metrics)


if __name__ == "__main__":
    main()
