import numpy as np
from common import *
from orientation import generateDirections, computeFilters, computeFiberOrientationFFT   
from plot import *
from imageio import *
from fit import *
from resample import *
from volumeio import *


def volumeProcessing():
    knit_path = r't' #   uint16, [256, 1014, 992]
    ds_path = r't'
    qs_path = r't'
    J_path = r't'
    orientation_path = r't'
    density_path = r''
    orientation_before_eJ_path = r''


    s = 1
    t = 2
    h = 16
    ed = 0.15
    eJ = -40

    nx = 32
    ny = 32
    nz = 6

    print 'load volume ..'
    v = load(knit_path).astype('float32')
    v = v.swapaxes(0, 1).swapaxes(1, 2)
    v = v[:,:,0:215]
    print 'volume shape: ', v.shape

    # denoise
    print 'denoising using ed ..'
    v[v < ed * 65535.0] = 0.0

    # compute orientation
    print 'binarize ..'
    vo = v.copy()
    fiber = vo > 0.0
    background = np.logical_not(fiber)
    vo[fiber] = 0.0
    vo[background] = 1.0

    if not os.path.exists(ds_path):
        print 'compute ds and qs: ..'
        ds = generateDirections(nx, ny, nz)
        qs = computeFilters(h, s, t, ds)
        print 'dump ds and qs ..'
        dump(ds_path, ds)
        dump(qs_path, qs)
    else:
        print 'load ds and qs ..'
        ds = load(ds_path)
        qs = load(qs_path)

    print 'compute orientation ..'
    start = time_current()
    J_max, d_max = computeFiberOrientationFFT(vo, ds, qs)
    print 'time used: ', time_current() - start

    del vo, ds, qs

    background2 = J_max < eJ
    background = np.logical_or(background, background2)
    del background2

    print 'denoising using eJ ..'
    v[background] = 0.0
    d_max_before_denoise = d_max.copy()
    d_max[background] = 0.0
    
    del fiber, background

    print 'dump density, J, orientation ..'
    dump(density_path, v)
    dump(J_path, J_max)
    dump(orientation_path, d_max)
    dump(orientation_before_eJ_path, d_max_before_denoise)


def createMitsubaVolume():
    density_path = r''
    orientation_path = r''
    density_path_mis = r''
    orientation_path_mis = r''

#    print 'create density mitsuba volume ...'
#    density = load(density_path)
#    print 'density shape: ', density.shape
#    createMitsubaGridVolumeSimple(density, density_path_mis, 0.15)

    print 'create orientation mitsuba volume ...'
    orientation = load(orientation_path)
    print 'orientation shape: ', orientation.shape
    createMitsubaGridVolumeSimple(orientation, orientation_path_mis, 0.15)

    print 'Done.'


def createVolumeFromTiffs():
    knit_images_folder = r''
    knit = readTiffsInFileFolder(knit_images_folder)
    dump(r'', knit)


if __name__ == '__main__':
    createMitsubaVolume()
