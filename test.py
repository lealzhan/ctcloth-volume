import numpy as np
from scipy import signal


def testConvolveFFT():
    a = np.array([[1, 2, 0, 0],
                 [5, 3, 0, 4],
                 [0, 0, 0, 7],
                 [9, 3, 0, 0]], dtype='float32')
    
    k = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype='float32')
    
    r = signal.fftconvolve(a, k, mode='same')
    print r


def testConvolveFFT2():
    f = np.arange(0, 27).reshape((3,3,3))
    q = np.zeros((3,3,3))
    q[1,1,1] = 1
    print f
    print q

    q_shape = (3, 3, 3)
    flip_q = [slice(None, None, -1)] * len(q_shape)
    k = q[flip_q]

    r = signal.fftconvolve(f, k, mode='same')
    print r


if __name__ == '__main__':
    testConvolveFFT2()
