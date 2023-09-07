from time import perf_counter
from typing import Optional

from colorama import Fore

from mjregrasping.grasp_and_settle import deactivate_release_and_moving, grasp_and_settle
from mjregrasping.move_to_joint_config import pid_to_joint_configs

from mjregrasping.movie import MjMovieMaker
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.real_val import RealValCommander
from mjregrasping.regrasp_planner_utils import get_geodesic_dist
from mjregrasping.rrt import GraspRRT
from mjregrasping.viz import Viz
from moveit_msgs.msg import MoveItErrorCodes


class BaseOnStuckMethod:

    def __init__(self, scenario, skeletons, goal, grasp_goal, grasp_rrt: GraspRRT):
        self.scenario = scenario
        self.skeletons = skeletons
        self.goal = goal
        self.grasp_goal = grasp_goal
        self.grasp_rrt = grasp_rrt

    def method_name(self):
        raise NotImplementedError()

    def on_stuck(self, phy: Physics, viz: Viz, mov: Optional[MjMovieMaker], val_cmd: Optional[RealValCommander] = None):
        raise NotImplementedError()

    def execute_best_grasp(self, best_grasp, mov, phy, viz):
        if best_grasp.res.error_code.val == MoveItErrorCodes.SUCCESS:
            viz.viz(best_grasp.phy, is_planning=True)
            print(f"Changing from {best_grasp.initial_locs} to {best_grasp.locs}")
            deactivate_release_and_moving(phy, best_grasp.strategy, viz, is_planning=False, mov=mov)
            pid_to_joint_configs(phy, best_grasp.res, viz, is_planning=False, mov=mov)
            grasp_and_settle(phy, best_grasp.locs, viz, is_planning=False, mov=mov)
            self.grasp_goal.set_grasp_locs(best_grasp.locs)
        else:
            print(Fore.RED + "Failed to find a plan." + Fore.RESET)


class OnStuckOurs(BaseOnStuckMethod):

    def __init__(self, scenario, skeletons, goal, grasp_goal, grasp_rrt: GraspRRT):
        super().__init__(scenario, skeletons, goal, grasp_goal, grasp_rrt)
        from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner
        self.planner = HomotopyRegraspPlanner(goal.loc, self.grasp_rrt, skeletons)

    def method_name(self):
        return "\\signature{}"

    def on_stuck(self, phy, viz, mov, val_cmd: Optional[RealValCommander] = None):
        initial_geodesic_dist = get_geodesic_dist(self.grasp_goal.get_grasp_locs(), self.goal.loc)
        planning_t0 = perf_counter()
        sim_grasps = self.planner.simulate_sampled_grasps(phy, viz, viz_execution=True)
        best_grasp = self.planner.get_best(sim_grasps, viz=viz)
        new_geodesic_dist = get_geodesic_dist(best_grasp.locs, self.goal.loc)
        # if we are unable to improve by grasping closer to the keypoint, update the blacklist and replan
        new_dist_is_lower = new_geodesic_dist < initial_geodesic_dist - hp['grasp_loc_diff_thresh']
        print(f"{new_geodesic_dist=:.3f} {initial_geodesic_dist=:.3f}: {new_dist_is_lower=}")
        if not new_dist_is_lower:
            print(Fore.YELLOW + "Updating blacklist and replanning..." + Fore.RESET)
            self.planner.update_blacklists(phy)
            best_grasp = self.planner.get_best(sim_grasps, viz=viz)
        self.planner.planning_times.append(perf_counter() - planning_t0)
        self.execute_best_grasp(best_grasp, mov, phy, viz)


class OnStuckTamp(BaseOnStuckMethod):

    def __init__(self, scenario, skeletons, goal, grasp_goal, grasp_rrt: GraspRRT):
        super().__init__(scenario, skeletons, goal, grasp_goal, grasp_rrt)
        from mjregrasping.tamp_regrasp_planner import TAMPRegraspPlanner
        self.planner = TAMPRegraspPlanner(scenario, goal, self.grasp_rrt, skeletons)

    def method_name(self):
        return f"Tamp{hp['tamp_horizon']}"

    def on_stuck(self, phy, viz, mov, val_cmd: Optional[RealValCommander] = None):
        planning_t0 = perf_counter()
        sim_grasps = self.planner.simulate_sampled_grasps(phy, viz, viz_execution=False)
        best_grasp = self.planner.get_best(sim_grasps, viz=viz)
        self.planner.planning_times.append(perf_counter() - planning_t0)
        self.execute_best_grasp(best_grasp, mov, phy, viz)


class OnStuckAlwaysBlacklist(OnStuckOurs):

    def method_name(self):
        return "Always Blacklist"

    def on_stuck(self, phy, viz, mov, val_cmd: Optional[RealValCommander] = None):
        planning_t0 = perf_counter()
        sim_grasps = self.planner.simulate_sampled_grasps(phy, viz, viz_execution=True)
        print("Blacklisting")
        self.planner.update_blacklists(phy)
        best_grasp = self.planner.get_best(sim_grasps, viz=viz)
        self.planner.planning_times.append(perf_counter() - planning_t0)
        self.execute_best_grasp(best_grasp, mov, phy, viz)
