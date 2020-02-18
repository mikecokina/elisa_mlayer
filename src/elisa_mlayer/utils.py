import numpy as np


def convert_to_numpy_array(iterable):
    return [np.array(record) for record in iterable]
