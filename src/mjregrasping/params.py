import numpy as np
from mjregrasping.cfg import ParamsConfig

from dynamic_reconfigure.encoding import Config
from dynamic_reconfigure.server import Server

# These values are considered const by convention
hp = {
    "regrasp_horizon":                    15,
    "regrasp_n_samples":                  24,
    "regrasp_temp":                       0.5,
    "n_g":                                2,
    "weight_activation_thresh":           0.95,
    "weight_deactivation_thresh":         0.05,
    "cost_activation_thresh":             0.25,
    "grasp_weight":                       20.0,
    "smoothness_weight":                  1.0,
    "controllability_weight":             1.0,
    "finger_weight":                      1.0,
    "finger_q_open":                      0.30,
    "finger_q_closed":                    0.15,
    "pull_cost_weight":                   0.005,
    "unstable_weight":                    100,
    "state_history_size":                 10,
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
    "horizon":                            15,
    "temp":                               0.1,
    "min_nongrasping_rope_gripper_dists": 0.5,
    "nongrasping_close":                  0.1,
    "near_threshold":                     0.02,
    "gripper_dfield":                     0.01,
    "action":                             0.01,
    "rope_motion_weight":                 0.1,
    "ever_not_grasping":                  10,
    "grasp_change_error":                 50,
    "thread_dir_weight":                  1.0,  # for threading
    "thread_orient_weight":               0.0,  # for threading
    "nongrasping_home":                   0.05,
    "contact_exponent":                   0.5,
    "max_contact_cost":                   1,
    "contact_cost":                       3.0,
    "frac_max_dq":                        0.5,
    "q_joint_weight":                     0.5,
    "grasp_goal_radius":                  0.03,
}


class Params:

    def __init__(self):
        self.srv = Server(ParamsConfig, self.callback)
        self.config = self.srv.update_configuration({})

    def callback(self, new_config: Config, _):
        self.config = new_config
        return new_config

    def update(self):
        self.config = self.srv.update_configuration(self.config)

    def __getattr__(self, item):
        return getattr(self.config, item)
