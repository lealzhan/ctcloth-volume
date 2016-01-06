import numpy as np
from common import load, dump, time_current
from orientation import generateDirections, computeFilters, computeFiberOrientationFFT
from plot import *
from imageio import *
from fit import *
from resample import *
from volumeio import *


def volumeProcessing():
    silk_path = r'D:\Dataset\round2\silk\silk.dat'
    ds_path = r'D:\Dataset\round2\silk\ds.dat'
    qs_path = r'D:\Dataset\round2\silk\qs.dat'
    J_path = r'D:\Dataset\round2\silk\J.dat'
    orientation_path = r'D:\Dataset\round2\silk\silk_orientation.dat'
    density_path = r'D:\Dataset\round2\silk\silk_density.dat'

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
    
    v = v[:,:,25:85]
    print 'volume shape: ', v.shape

    # denoise
    print 'denoising using ed ...'
    v[v < ed * 65535.0] = 0.0

    # fit
    print 'fit volume ...'
    vf = v.copy()
    fiber = vf > 0.0
    background = np.logical_not(fiber)
    vf[fiber] = 1.0
    vf[background] = 0.0

    m = poly2ndfitVolume(vf)
    X, Y, Z = generatePoly2ndSurfaceSimple(m, vf.shape[0], vf.shape[1])
    del vf

    # resample
    print 'resample volume ...'
    v = resampleVolume(v, Z)
    print 'volume shape after resampling: ', v.shape
    del X, Y, Z

    # compute orientation 
    v = v[:,:,22:62]
    print 'volume shape for computing orientation: ', v.shape

    # binarize volume for filtering
    # 0: fiber, 1: background
    print 'binarize ...'
    vo = v.copy()
    fiber = vo > 0.0
    background = np.logical_not(fiber)
    vo[fiber] = 0.0
    vo[background] = 1.0

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
    J_max, d_max = computeFiberOrientationFFT(vo, ds, qs)
    print 'time used: ', time_current() - start

    del vo

    background2 = J_max < eJ
    background = np.logical_or(background, background2)

    print 'denoising using eJ ...'
    v[background] = 0.0
    d_max_before_denoise = d_max.copy()
    d_max[background] = 0.0

    print 'dump density, J, orientation ...'
    dump(density_path, v)
    dump(J_path, J_max)
    dump(orientation_path, d_max)
    dump(orientation_path + '_before_denoise', d_max_before_denoise)


def createMitsubaVolume():
    density_path = r'D:\Dataset\round2\silk\silk_density.dat'
    orientation_path = r'D:\Dataset\round2\silk\silk_orientation.dat'
    density_path_mis = r'D:\Dataset\round2\silk\silk_density_mis_xy200-800.vol'
    orientation_path_mis = r'D:\Dataset\round2\silk\silk_orientation_mis_xy200-800.vol'

    print 'create density mitsuba volume ...'
    density = load(density_path)
    density = density[200:800, 200:800]
    print 'density shape: ', density.shape
    createMitsubaGridVolumeSimple(density, density_path_mis, 0.5)

    print 'create orientation mitsuba volume ...'
    orientation = load(orientation_path)
    orientation = orientation[200:800, 200:800]
    print 'orientation shape: ', orientation.shape
    createMitsubaGridVolumeSimple(orientation, orientation_path_mis, 0.5)

    print 'Done.'


if __name__ == '__main__':
    createMitsubaVolume()
