import pickle

import mujoco

from mjregrasping.physics import Physics


def save_data_and_eq(phy: Physics):
    data_and_eq = {
        'data': phy.d,
        'eq':   {},
    }
    for eq_idx in range(phy.m.neq):
        eq = phy.m.eq(eq_idx)
        data_and_eq['eq'][eq_idx] = {
            'active': eq.active.copy(),
            'data':   eq.data.copy(),
            'obj2id': eq.obj2id.copy(),
        }
    with open("data_and_eq.pkl", "wb") as f:
        pickle.dump(data_and_eq, f)


def load_data_and_eq(m: mujoco.MjModel, forward=True):
    """ Modifies the model in place and returns a new instance of MjData """
    with open("data_and_eq.pkl", "rb") as f:
        data_and_eq = pickle.load(f)
    for eq_idx in range(m.neq):
        eq = m.eq(eq_idx)
        eq.active[:] = data_and_eq['eq'][eq_idx]['active']
        eq.data[:] = data_and_eq['eq'][eq_idx]['data']
        eq.obj2id[:] = data_and_eq['eq'][eq_idx]['obj2id']
    # NOTE: by doing this, we change the data. That means if you want perfect reproducibility you should skip this
    if forward:
        mujoco.mj_forward(m, data_and_eq['data'])
    return data_and_eq['data']
