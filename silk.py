import numpy as np
from common import load, dump, time_current
from orientation import generateDirections, computeFilters, computeFiberOrientationFFT
from plot import *
from imageio import *


def main():
    silk_path = r'D:\Dataset\round2\silk\silk_density.dat'
    ds_path = r'D:\Dataset\round2\silk\ds.dat'
    qs_path = r'D:\Dataset\round2\silk\qs.dat'
    J_path = r'D:\Dataset\round2\silk\J.dat'
    d_path = r'D:\Dataset\round2\silk\silk_orientation.dat'

    s = 3
    t = 4
    h = 12
    ed = 0.4
    eJ = -6

    nx = 32
    ny = 32
    nz = 6

    print 'load volume ...'
    v = load(silk_path)
    
    v= v[400:600, 400:600, 55:75]
    print 'volume shape: ', v.shape

    # denoise
    v[v < ed * 65535.0] = 0.0

    # binarize volume for filtering
    # 0: fiber, 1: background
    fiber = v > 0.0
    background = np.logical_not(fiber)
    v[fiber] = 0.0
    v[background] = 1.0

    # fit (not implement yet)

    if not os.path.exists(ds_path):
        print 'compute ds and qs ...'
        ds = generateDirections(nx, ny, nz)
        qs = computeFilters(h, s, t, ds)
        print 'dump ds and qs ...'
        dump(ds_path, ds)
        dump(qs_path, qs)
    else:
        print 'load ds and qs ...'
        ds = load(ds_path)
        qs = load(qs_path)
    
    print 'compute orientation ...'
    start = time_current()    
    J_max, d_max = computeFiberOrientationFFT(v, ds, qs)
    print 'time used: ', time_current() - start

    # ..
#    d_max[background] = 0.0

    print 'dump J, orientation ...'
    dump(J_path, J_max)
    dump(d_path, d_max)


if __name__ == '__main__':
    main()
