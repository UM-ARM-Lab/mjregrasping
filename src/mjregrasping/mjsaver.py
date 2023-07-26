import pickle
from pathlib import Path

import mujoco

from mjregrasping.physics import Physics

DEFAULT_PATH = Path("states/data_and_eq.pkl")


def save_data_and_eq(phy: Physics, path=DEFAULT_PATH):
    data_and_eq = {
        'data': phy.d,
        'eq':   {},
    }
    for eq_idx in range(phy.m.neq):
        eq = phy.m.eq(eq_idx)
        data_and_eq['eq'][eq_idx] = {
            'name': eq.name,
            'active': eq.active.copy(),
            'data':   eq.data.copy(),
            'obj2id': eq.obj2id.copy(),
        }
    with path.open("wb") as f:
        pickle.dump(data_and_eq, f)


def load_data_and_eq(m: mujoco.MjModel, path=DEFAULT_PATH, forward=True):
    """ Modifies the model in place and returns a new instance of MjData """
    with path.open("rb") as f:
        data_and_eq = pickle.load(f)
    for eq_idx in range(m.neq):
        eq = m.eq(eq_idx)
        for eq_idx, eq_value in data_and_eq['eq'].items():
            if eq.name == eq_value['name']:
                eq.active[:] = eq_value['active']
                eq.data[:] = eq_value['data']
                eq.obj2id[:] = eq_value['obj2id']
    # NOTE: by doing this, we change the data. That means if you want perfect reproducibility you should skip this
    if forward:
        mujoco.mj_forward(m, data_and_eq['data'])
    return data_and_eq['data']
