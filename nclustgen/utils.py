import numpy as np


def tensor_value_check(value):

    try:
        return float(value.replace(",", "."))

    except ValueError:
        return np.nan


def loader(module, cls):
    return getattr(module, cls) if isinstance(cls, str) else cls
