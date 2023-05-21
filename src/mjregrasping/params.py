from mjregrasping.cfg import ParamsConfig

from dynamic_reconfigure.encoding import Config
from dynamic_reconfigure.server import Server

# These values are considered const by convention
hp = {
    "max_perf":                           False,
    "iters":                              5,
    "max_move_to_goal_iters":             200,
    "max_grasp_iters":                    100,
    "max_plan_to_goal_iters":             15,
    "max_grasp_plan_iters":               15,
    "warmstart":                          5,
    "move_sub_time_s":                    0.1,
    "grasp_sub_time_s":                   0.05,
    "plan_sub_time_s":                    0.2,
    "plan_settle_steps":                  5,
    "settle_steps":                       20,
    "num_samples":                        50,
    "horizon":                            9,
    "n_samples":                          50,
    "lambda":                             0.02,
    "min_nongrasping_rope_gripper_dists": 0.5,
    "nongrasping_close":                  0.1,
    "near_threshold":                     0.02,
    "gripper_dfield":                     0.01,
    "action":                             0.01,
    "nongrasping_home":                   0.05,
    "contact_exponent":                   0.5,
    "max_contact_cost":                   1,
    "contact_cost":                       5.0,
    "f_eq_err_weight":                    5.0,
    "f_grasp_weight":                     0.1,
    "f_goal_weight":                      0.8,
    "running_cost_weight":                0.02,
    "f_settle_weight":                    0.1,
    "point_dist_weight":                  2.0,
    "needs_regrasp_again":                10,
    "frac_max_dq":                        0.2,
}


class Params:

    def __init__(self):
        self.srv = Server(ParamsConfig, self.callback)
        self.config = self.srv.update_configuration({})

    def callback(self, new_config: Config, _):
        self.config = new_config
        return new_config

    def __getattr__(self, item):
        return getattr(self.config, item)
