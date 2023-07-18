# noinspection PyUnresolvedReferences
from mjregrasping.cfg import ParamsConfig

from dynamic_reconfigure.encoding import Config
from dynamic_reconfigure.server import Server

# These values are considered const by convention
hp = {
    "regrasp_horizon":          15,
    "regrasp_n_samples":        50,
    "regrasp_temp":             0.15,
    "goal_weight":              1.0,
    "grasp_pos_weight":         1.0,
    "grasp_near_weight":        5.0,
    "grasp_finger_weight":      0.5,
    "grasp_loc_diff_thresh":    0.07,  # A percentage of rope length
    "smoothness_weight":        0.02,
    "home_weight":              0.005,
    "nearby_locs_weight":       10,
    "unstable_weight":          100,
    "ever_not_grasping_weight": 5.0,
    "finger_q_open":            0.3,
    "finger_q_closed":          0.08,
    "state_history_size":       8,
    "warmstart":                10,
    "settle_steps":             10,
    "thread_dir_weight":        1.0,  # for threading
    "thread_orient_weight":     0.0,  # for threading
    "contact_exponent":         0.5,
    "max_contact_cost":         1,
    "max_expected_contacts":    6,
    "contact_cost":             3.0,
    "contact_force_weight":     0.0006,
    "settle_weight":            100.0,
    "eq_err_weight":            500.0,
    "frac_max_dq":              0.5,
    "q_joint_weight":           0.5,
    "frac_dq_threshold":        0.2,
    "grasp_goal_radius":        0.045,
    "sub_time_s":               0.15,
    "min_sub_time_s":           0.05,
    "max_sub_time_s":           0.30,
    "geodesic_weight":          25.0,
    "sim_ik_nstep":             20,
    "sim_ik_sub_time_s":        0.25,
    "sim_ik_solref_decay":      0.8,
    "sim_ik_min_solref":        0.02,
    "attach_loop_size":         0.02,
    "bayes_opt":                {
        "n_iter": 20,
        "n_init": 10,
    }
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
