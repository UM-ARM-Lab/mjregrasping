import pickle


def save_data_and_eq(model, data):
    data_and_eq = {
        'data': data,
        'eq':   {},
    }
    for eq_idx in range(model.neq):
        eq = model.eq(eq_idx)
        data_and_eq['eq'][eq_idx] = {
            'active': eq.active.copy(),
            'data':   eq.data.copy(),
        }
    with open("data_and_eq.pkl", "wb") as f:
        pickle.dump(data_and_eq, f)


def load_data_and_eq(model):
    """ Modifies the model in place and returns a new instance of MjData """
    with open("data_and_eq.pkl", "rb") as f:
        data_and_eq = pickle.load(f)
    for eq_idx in range(model.neq):
        eq = model.eq(eq_idx)
        eq.active[:] = data_and_eq['eq'][eq_idx]['active']
        eq.data[:] = data_and_eq['eq'][eq_idx]['data']
    return data_and_eq['data']
