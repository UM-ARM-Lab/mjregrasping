params = {
    'iters':            50,
    'warmstart':        5,
    'needs_regrasp': {
        'min_dq': 0.05,
        'min_command': 0.1,
    },
    'move_to_goal':     {
        'horizon':   9,
        'n_samples': 50,
        'lambda':    0.02,
    }
}
