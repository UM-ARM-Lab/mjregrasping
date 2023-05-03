params = {
    'iters':            50,
    'warmstart':        5,
    'needs_regrasp':    {
        'min_dq':      0.020,
        'min_command': 0.55,
    },
    'move_to_goal':     {
        'horizon':   9,
        'n_samples': 50,
        'lambda':    0.02,
    },
    'costs':            {
        'min_nongrasping_rope_gripper_dists': 0.0,
        'nongrasping_close':                  0.15,
        'no_points_useful':                   10,
        'cannot_progress':                    1000,
        'near_threshold':                     0.02,
        'contact_exponent':                   0.5,
        'max_contact_cost':                   1.0,
        'gripper_dir':                        1.0,
    },
    'move_sub_time_s':  0.1,
    'grasp_sub_time_s': 0.02,
}
