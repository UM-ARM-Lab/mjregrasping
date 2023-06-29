import numpy as np


def softmax(x, temp, sub_max=True):
    if sub_max:
        x = x - x.max()
    x = x / temp
    z = np.exp(x)
    return z / z.sum()
