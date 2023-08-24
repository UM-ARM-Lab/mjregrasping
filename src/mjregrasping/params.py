# noinspection PyUnresolvedReferences
from mjregrasping.cfg import ParamsConfig

from dynamic_reconfigure.encoding import Config
from dynamic_reconfigure.server import Server

# These values are considered const by convention
hp = {
    "horizon":                         15,
    "n_samples":                       50,
    "temp":                            0.15,
    "keypoint_weight":                 1.0,
    "next_xpos_sdf_weight":            0.1,
    "weifu_sdf_weight":                1.0,  # For baseline
    "torso_weight":                    0.1,
    "grasp_pos_weight":                1.0,
    "grasp_near_weight":               1.0,
    "grasp_finger_weight":             0.5,
    "grasp_loc_diff_thresh":           0.025,  # A percentage of rope length
    "smoothness_weight":               0.02,
    "angle_cost_weight":               0.2,  # for ThreadingGoal
    "thread_geodesic_w":               20.0,
    "nearby_locs_weight":              10,
    "unstable_weight":                 100,
    "ever_not_grasping_weight":        5.0,
    "finger_q_open":                   0.3,
    "finger_q_closed":                 0.06,
    "finger_q_pregrasp":               0.12,
    "state_history_size":              8,
    "warmstart":                       8,
    "settle_steps":                    10,
    "contact_exponent":                0.5,
    "nongrasping_rope_contact_weight": 0.5,
    "gripper_to_goal_weight":          0.2,
    "robot_dq_weight":                 0.3,
    "max_contact_cost":                1,
    "max_expected_contacts":           6,
    "contact_cost":                    3.0,
    "rope_disp_weight":                25.0,
    "eq_err_weight":                   400.0,
    "frac_max_dq":                     0.5,
    "q_joint_weight":                  0.5,
    "frac_dq_threshold":               0.3,  # TODO: lower this back to 0.2
    "max_max_dq":                      0.008,
    "grasp_goal_radius":               0.045,
    "sub_time_s":                      0.20,
    "min_sub_time_s":                  0.10,
    "max_sub_time_s":                  0.30,
    "geodesic_weight":                 30.0,
    "first_order_weight":              10.0,
    "sim_ik_nstep":                    25,
    "sim_ik_sub_time_s":               0.25,
    "sim_ik_solref_decay":             0.8,
    "sim_ik_min_solref":               0.02,
    "attach_loop_size":                0.02,
    "joint_kp":                        6.0,
    "act_windup_limit":                0.02,
    "bayes_opt":                       {
        "n_iter": 14,
        "n_init": 6,
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
