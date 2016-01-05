import numpy as np


def poly2ndfit(x, y, z, w=None):
    c = np.ones(x.size, dtype='float32')
    G = np.column_stack((x**2, y**2, x*y, x, y, c))

    if w is not None:
        G *= w[:, np.newaxis]
        z *= w

    m, _, _, _ = np.linalg.lstsq(G, z)
    return m


def poly2ndval(x, y, m):
    return x**2 * m[0] + y**2 * m[1] + x * y * m[2] + x * m[3] + y * m[4] + m[5]


def poly2ndfitVolume(v):
    nx, ny, nz = v.shape
    x = np.arange(0, nx)
    y = np.arange(0, ny)
    z = np.arange(0, nz)

    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    del x, y, z

    m = poly2ndfit(xv.ravel(), yv.ravel(), zv.ravel(), w=v.ravel())    
    return m


def generatePoly2ndSurfaceSimple(m, nx, ny):
    m = np.asarray(m)
    x = np.arange(0, nx)
    y = np.arange(0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = poly2ndval(X, Y, m)

    return (X, Y, Z)


if __name__ == '__main__':
    pass
