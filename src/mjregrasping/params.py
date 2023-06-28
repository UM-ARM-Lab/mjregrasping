from mjregrasping.cfg import ParamsConfig

from dynamic_reconfigure.encoding import Config
from dynamic_reconfigure.server import Server

# These values are considered const by convention
hp = {
    "regrasp_horizon":        15,
    "regrasp_n_samples":      32,
    "regrasp_temp":           0.1,
    "n_g":                    2,
    "goal_weight":            1.0,
    "grasp_weight":           20.0,
    "regrasp_weight":         1.0,
    "finger_weight":          0.25,
    "smoothness_weight":      0.1,
    "regrasp_near_weight":    0.1,
    "controllability_weight": 1.0,
    "unstable_weight":        100,
    "finger_q_open":          0.4,
    "finger_q_closed":        0.08,
    "state_history_size":     10,
    "warmstart":              10,
    "sub_time_s":             0.1,
    "settle_steps":           20,
    "num_samples":            24,
    "horizon":                15,
    "temp":                   0.1,
    "ever_not_grasping":      5,
    "grasp_change_error":     50,
    "thread_dir_weight":      1.0,  # for threading
    "thread_orient_weight":   0.0,  # for threading
    "nongrasping_home":       0.05,
    "contact_exponent":       0.5,
    "max_contact_cost":       1,
    "contact_cost":           3.0,
    "frac_max_dq":            0.5,
    "q_joint_weight":         0.5,
    "grasp_goal_radius":      0.045,
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
