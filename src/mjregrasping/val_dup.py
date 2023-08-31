import numpy as np


def val_dup(x):
    """ duplicate index 9 and 17 """
    x_dup = np.insert(x, 10, x[9])
    x_dup = np.insert(x_dup, 19, x[17])
    return x_dup
