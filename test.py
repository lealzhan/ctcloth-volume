import numpy as np
from scipy import signal, ndimage
from common import time_current


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


def testTime():
    a = np.ones((212,212,32), dtype='float32') * 0.1 
    k = np.ones((12,12,12), dtype='float32') * 1.1

    start = time_current()
    r = signal.fftconvolve(a, k, mode='same')
    print time_current() - start

    del r

    start = time_current()
    r1 = ndimage.correlate(a, k, mode='constant', cval=1.0)
    print time_current() - start


if __name__ == '__main__':
    testTime()
