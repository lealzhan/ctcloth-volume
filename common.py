import numpy as np


def load(file_name, mode='rb'):
    with open(file_name, mode) as file:
        data = np.load(file)
    return data


def dump(file_name, data, mode='wb'):
    with open(file_name, mode) as file:
        data.dump(file)

if __name__ == '__main__':
	pass
