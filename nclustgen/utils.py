import numpy as np


def tensor_value_check(value):

    try:
        return float(value)

    except ValueError:
        return np.nan
