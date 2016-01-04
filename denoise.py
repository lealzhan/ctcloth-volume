import numpy as np


def denoise(data, ed, value=0.0):
    data[data < ed] = value


if __name__ == '__main__':
    pass
