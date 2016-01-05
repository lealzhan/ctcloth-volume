import numpy as np


def resampleVolume(v, z):
    zmin = z.min()
    z = z - zmin
    zmax = z.max()

    nx, ny, nz = v.shape
    x = np.arange(0, nx)
    y = np.arange(0, ny)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    rv = np.zeros((nx, ny, nz + int(zmax+0.5)), dtype='float32')

    for (i, j) in zip(xv.ravel(), yv.ravel()):
        s = int(zmax + 0.5) - int(z[i, j] + 0.5)
        rv[i, j, s:s+nz] = v[i, j, :]

    return rv


if __name__ == '__main__':
    pass