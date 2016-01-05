import numpy as np
from scipy import signal, ndimage
from common import time_current, load
from plot import *
from fit import *


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


def testFit():
    print 'load volume ...'
    v_path = r''
    v = load(v_path)    # float32, [0, 65535]
    print 'volume shape: ', v.shape

    # first, denoise using ed, and eJ?!
    print 'denoise using ed ...'
    bg = [v < 0.4 * 65535]
    v[bg] = 0.0

    # need binarize?
    print 'binarize ...'
    fiber = np.logical_not(bg)
    v[fiber] = 1.0
    
    m = poly2ndfitVolume(v)
    X, Y, Z = generatePoly2ndSurfaceSimple(m, v.shape[0], v.shape[1])
    plotSurface(X, Y, Z)


if __name__ == '__main__':
    print 'load volume ...'
    v_path = r'D:\Dataset\round2\silk\silk_density.dat' # 1013, 992, 105
    v = load(v_path)    # float32, [0, 65535]
    v = v[:,:,25:85]
    print 'volume shape: ', v.shape

    # first, denoise using ed, and eJ?!
    print 'denoise using ed ...'
    bg = (v < 0.4 * 65535)
    v[bg] = 0.0

    # need binarize?
    print 'binarize ...'
    fiber = np.logical_not(bg)
    v[fiber] = 1.0
    
    m = poly2ndfitVolume(v)
    X, Y, Z = generatePoly2ndSurfaceSimple(m, v.shape[0], v.shape[1])
    plotSurface(X, Y, Z)
