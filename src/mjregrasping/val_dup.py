import numpy as np


def val_dup(x):
    """ duplicate index 9 and 17 """
    x_dup = np.insert(x, 10, x[9])
    x_dup = np.insert(x_dup, 19, x[17])
    return x_dup


def val_dedup(x):
    """ removes the duplicated gripper values, which are at indices 10 and 19 """
    return np.concatenate([x[:10], x[11:19], x[20:]])
