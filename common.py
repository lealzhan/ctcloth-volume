import numpy as np
import timeit


def load(file_name, mode='rb'):
    with open(file_name, mode) as file:
        data = np.load(file)
    return data


def dump(file_name, data, mode='wb'):
    with open(file_name, mode) as file:
        data.dump(file)


def time_current():
    return timeit.default_timer()


if __name__ == '__main__':
	pass
