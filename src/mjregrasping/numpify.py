import numpy as np
import torch

import genpy


def numpify(x, dtype=np.float32):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, list):
        if len(x) == 0:
            return np.array(x)
        if isinstance(x[0], int):
            return np.array(x, dtype=dtype)
        elif isinstance(x[0], float):
            return np.array(x, dtype=dtype)
        elif isinstance(x[0], str):
            return np.array(x, dtype=np.str)
        else:
            l = [numpify(xi) for xi in x]
            # NOTE: if l is list of dicts for instance, we don't want to convert to an array.
            #  But if it's a list of lists (e.g. array) we do convert, so this is how we test for that
            l_arr = np.array(l)
            if l_arr.dtype in [np.float32, np.float64, np.int32, np.int64]:
                return l_arr
            else:
                return l
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, dict):
        return {k: numpify(v) for k, v in x.items()}
    elif isinstance(x, tuple):
        return tuple(numpify(x_i) for x_i in x)
    elif isinstance(x, int):
        return x
    elif isinstance(x, bytes):
        return x
    elif isinstance(x, str):
        return x
    elif isinstance(x, float):
        return x
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, np.float32):
        return x
    elif isinstance(x, np.int64):
        return x
    elif isinstance(x, np.int32):
        return x
    elif isinstance(x, np.bool_):
        return x
    elif isinstance(x, np.bytes_):
        return x
    elif x is None:
        return None
    elif isinstance(x, genpy.Message):
        return x
    else:
        raise NotImplementedError(type(x))
