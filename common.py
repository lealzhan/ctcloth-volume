import numpy as np


def load(file_name, mode='rb'):
    with open(file_name, mode) as file:
        data = np.load(file)
    return data


def save(file_name, data, mode='wb'):
    with open(file_name, mode) as file:
        np.save(file, data)