# calculate fiber orientation
import numpy as np
from scipy import ndimage
from scipy import signal


def generateBoundingPoints(nx, ny, nz):
    x = np.arange(0, nx)
    y = np.arange(0, ny)
    z = np.arange(0, nz)
    xm, ym, zm = np.meshgrid(x, y, z, indexing='ij')

    xm_l = (xm == 0)        # left
    xm_r = (xm == nx - 1)   # right
    xm_bound = np.logical_or(xm_l, xm_r)

    ym_b = (ym == 0)        # back
    ym_f = (ym == ny - 1)   # front
    ym_bound = np.logical_or(ym_b, ym_f)

    zm_b = (zm == 0)        # bottom
    zm_u = (zm == nz - 1)   # up
    zm_bound = np.logical_or(zm_b, zm_u)

    bound = np.logical_or(xm_bound, ym_bound)
    bound = np.logical_or(bound, zm_bound)

    xb = xm[bound]
    yb = ym[bound]
    zb = zm[bound]

    return np.column_stack((xb.ravel(), yb.ravel(), zb.ravel()))


def generateDirections(nx, ny, nz):
    bound_points = generateBoundingPoints(nx, ny, nz)
    center = np.array([nx / 2, ny / 2, nz / 2])

    directions = bound_points - center
    directions = directions.astype('float32')

    for i in range(0, directions.shape[0]):
        sqrt_sum = np.sqrt((directions[i]**2).sum())
        directions[i] /= sqrt_sum

    return directions


def computeFilter(h, s, t, d):
    '''d: unit direction'''
    c = np.array([h/2, h/2, h/2], dtype='float32')
    q = np.empty((h, h, h), dtype='float32')
    epsilon = 0.000001
    for (xi, yi, zi) in np.ndindex((h, h, h)):
        p = np.asarray([xi, yi, zi], dtype='float32')
        cp = p - c
        r_sq = np.abs((cp**2).sum() - (np.dot(cp, d))**2)
        if (cp**2).sum() < epsilon or r_sq < epsilon:
            r_sq = 0.0
        q[xi, yi, zi] = -2 * np.exp(-s * r_sq) + np.exp(-t * r_sq)


def computeFiberOrientation(v, ds, qs):
    neg_inf = -999999.0
    J_max = np.ones_like(v, dtype='float32') * neg_inf
    d_max = np.zeros_like((v.shape[0], v.shape[1], v.shape[2], 3), dtype='float32')

    for i in range(qs.shape[0]):
        J = ndimage.correlate(v, qs[i], mode='constant', cval=1.0)
        update = J > J_max
        J_max[update] = J[update]
        d_max[update] = ds[i]

    return (J_max, d_max)


def computeFiberOrientation2(v, ds, qs):
    '''fft based convolution'''
    # TODO:
    # r = signal.fftconvolve(a, k, mode='same')
    pass


if __name__ == '__main__':
    pass