from mjregrasping.cfg import ParamsConfig

from dynamic_reconfigure.encoding import Config
from dynamic_reconfigure.server import Server

# These values are considered const by convention
hp = {
    "n_g":                                2,
    "weight_activation_thresh":           0.95,
    "weight_deactivation_thresh":         0.05,
    "cost_activation_thresh":             0.25,
    "grasp_weight":                       5.0,
    "pull_cost_weight":                   0.005,
    "iters":                              5,
    "max_move_to_goal_iters":             600,
    "max_grasp_iters":                    100,
    "max_plan_to_goal_iters":             40,
    "warmstart":                          10,
    "move_sub_time_s":                    0.1,
    "grasp_sub_time_s":                   0.1,
    "plan_sub_time_s":                    0.1,
    "plan_settle_steps":                  20,
    "settle_steps":                       20,
    "num_samples":                        24,
    "num_samples_when_scoring":           10,
    "horizon":                            9,
    "n_samples":                          50,
    "lambda":                             0.02,
    "min_nongrasping_rope_gripper_dists": 0.5,
    "nongrasping_close":                  0.1,
    "near_threshold":                     0.02,
    "gripper_dfield":                     0.01,
    "action":                             0.01,
    "rope_motion_weight":                 0.1,
    "nongrasping_home":                   0.05,
    "contact_exponent":                   0.5,
    "max_contact_cost":                   1,
    "contact_cost":                       3.0,
    "f_eq_err_weight":                    4.0,
    "f_goal_weight":                      2.0,
    "f_final_goal_weight":                5.0,
    "f_settle_weight":                    0.25,
    "f_contact_weight":                   1.0,
    "pull_gripper_to_goal_solref":        0.2,
    "running_cost_weight":                0.02,
    "point_dist_weight":                  2.0,
    "needs_regrasp_again":                10,
    "frac_max_dq":                        0.2,
    "q_joint_weight":                     0.5,
    "cma_sigma":                          0.3,
    "cma_opts":                           {
        'popsize':   5,
        'seed':      1,
        'maxfevals': 25,
        'tolx':      1e-2,
        'bounds':    [0, 1],
    }
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
