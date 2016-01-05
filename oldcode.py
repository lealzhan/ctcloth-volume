# old code

# Util.py ---------------------------
import numpy as np
import matplotlib.pylab as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
import scipy.io

# --- load and dump data helper functions --- #
def loadData(file_path, mode='rb'):
    '''Load dumped binary file.
    '''
    with open(file_path, 'rb') as data_file:
        data = np.load(data_file)
        
    return data
    

def dumpData(data, file_path, mode='wb'):
    '''Dump data into a binary file.
    '''
    with open(file_path, 'wb') as data_file:
        data.dump(data_file)
# --------------------------------


# --- plot helper functions --- #
def plotImage(img, cm=plt.cm.gray):
    fig, ax = plt.subplots()
    pic = ax.imshow(img, cmap=cm)
    fig.colorbar(pic)
    fig.show()


def plotHist(data, bins):
    hist, bins = np.histogram(data, bins=bins)    
    fig, ax = plt.subplots()
    
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ax.bar(center, hist, align='center', width=width)  
    
    fig.show()

# --------------------------------


# --- histogram equalization --- #
def histEq(data, bins=np.arange(65537)):
    '''Histogram equalize the input data. Return histogram equalized result
    withe range between [0, 1].

    Parameters:
    data: array_like
        Input data. The hisogram is computed over the flattened array.
    bins: int or sequence of scalars, optional
        E.g, L, L is the image's maximum intensity
    range: (float, float), optional
        E.g, (0, L), L is the image's maximum intensity


    Returns:
    res: array_like
        The intensity is between [0, 1]
    '''
    # not normlized, since in default: density=False
    hist, bins = np.histogram(data.flatten(), bins=bins)
    cdf = hist.cumsum()
    cdf = cdf / (cdf[-1] * 1.0)

    res = np.interp(data.flatten(), bins[:-1], cdf)

    return res.reshape(data.shape)


# --- polynomial fitting of 3d data
def polyfit2d(x, y, z, order=2, w=None):
    '''Polynomial fitting 3D data, z = f(x, y).

    Parameters
    x: (N, ), array_like
    y: (N, ), array_like
    z: (N, ), array_like
    w: (N, ), array_like, optional
    
    Reference
    http://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent
    https://github.com/numpy/numpy/blob/v1.9.1/numpy/lib/polynomial.py#L396
    '''
#    x = np.asarray(x) + 0.0
#    y = np.asarray(y) + 0.0
#    z = np.asarray(z) + 0.0
    n_cols = (order + 1)**2
    G = np.zeros((x.size, n_cols))

    ij = itertools.product(range(order + 1), range(order + 1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j

    # Apply weighting
    if w is not None:
        #w = np.asarray(w) + 0.0
        G *= w[:, np.newaxis]
        z *= w

    m, _, _, _ = np.linalg.lstsq(G, z)
    return m


def polyval2d(x, y, m):
    order = int(np.round(np.sqrt(len(m)))) - 1
    ij = itertools.product(range(order + 1), range(order + 1))

    z = np.zeros_like(x)
    for a, (i, j) in zip(m, ij):
        z += a * x**i * y**j
    
    return z


def polyfit2dPure(x, y, z, order=2, w=None):
    '''
    References:
    http://pingswept.org/2009/06/15/least-squares-fit-of-a-surface-to-a-3d-cloud-of-points-in-python-(with-ridiculous-application)/
    '''
#    x = np.asarray(x) + 0.0
#    y = np.asarray(y) + 0.0
#    z = np.asarray(z) + 0.0

    deg = order + 1
    Gx = np.vander(x, deg)
    Gy = np.vander(y, deg)
    G = np.hstack((Gx, Gy))

    del x, y, Gx, Gy
    
    # Apply weighting
    if w is not None:
#        w = np.asarray(w) + 0.0
        G *= w[:, np.newaxis]
        z *= w
        
    del w

    m, _, _, _ = np.linalg.lstsq(G, z)
    return m


def polyval2dPure(x, y, m):
    xcoeffs = m[0:len(m)/2]
    ycoeffs = m[len(m)/2:len(m)]
    
    fx = np.poly1d(xcoeffs)
    fy = np.poly1d(ycoeffs)
    
    return fx(x) + fy(y)


def plotSurface(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    fig.show()


def poly2ndfit2d(x, y, z, w=None):
    c = np.ones(x.size, dtype='float32')
    G = np.column_stack((x**2, y**2, x*y, x, y, c))
 #   G = np.column_stack((x**2, y**2, x, y, c))

    if w is not None:
        G *= w[:, np.newaxis]
        z *= w
    
    m, _, _, _ = np.linalg.lstsq(G, z)
    
    return m
    
def poly2ndval2d(x, y, m):
    return x**2 * m[0] + y**2 * m[1] + x * y * m[2] + x * m[3] + y * m[4] + m[5]
   
    
def numpyToMat(file_path, mdict):
    '''Convert numpy to matlab .mat file.
    '''
    scipy.io.savemat(file_path, mdict=mdict)


def loadMat(file_path):
    '''Load matlab .mat file as numpy array.
    '''
    return scipy.io.loadmat(file_path)


if __name__ == '__main__':
    pass
# Util.py ---------------------------

# Test.py ---------------------------
import numpy as np
import struct
import Util


def loadFeltJ():
    j0 = r'D:\Document\clothSimulation\Dataset\felt\J0_100.dat'
    j1 = r'D:\Document\clothSimulation\Dataset\felt\J100_200.dat'
    j2 = r'D:\Document\clothSimulation\Dataset\felt\J200_300.dat'
    j3 = r'D:\Document\clothSimulation\Dataset\felt\J300_400.dat'
    j4 = r'D:\Document\clothSimulation\Dataset\felt\J400_500.dat'
    j5 = r'D:\Document\clothSimulation\Dataset\felt\J500_508.dat'
    
    j0 = Util.loadData(j0)
    j1 = Util.loadData(j1)
    j2 = Util.loadData(j2)
    j3 = Util.loadData(j3)
    j4 = Util.loadData(j4)
    j5 = Util.loadData(j5)

    j = np.concatenate((j0, j1, j2, j3, j4, j5), axis=0)

    return j


def test():
    a = np.ones((2,2), dtype='int32')


def test():
    a = np.ones((2,2), dtype='int32')


def test():
    a = np.ones((2,2), dtype='int32')


def test():
    a = np.ones((2,2), dtype='int32')


def test():
    a = np.ones((2,2), dtype='int32')


def test():
    a = np.ones((2,2), dtype='int32')

    a[0,1] = 2
    a[1,0] = 3
    a[1,1] = 4

    f_name = 'a.dat'
    with open(f_name, 'wb') as f:
        f.write(a.data)
        b = np.array([0.5], dtype='float32')
        f.write(b.data)
        c = 567
#        f.write(c) # error, must be string or buffer
        f.write(struct.pack('i', c))
        f.write('V')

def readTest():
    f_name = 'a.dat'
    with open(f_name, 'rb') as f:
        data = f.read(4 * 4)
        data = struct.unpack('@4i', data)
        
        print type(data)
        print data

        data = f.read(4)
        data = struct.unpack('@f', data)
        print type(data)
        print data

        data = f.read(4)
        data = struct.unpack('@i', data)
        print type(data)
        print data

        data = f.read(1)
        data = struct.unpack('@c', data)
        print type(data)
        print data

if __name__ == '__main__':
    pass
# Test.py ---------------------------

# Denoising.py ----------------------
import numpy as np
import Util


def denoise(d, J, ori, ed, eJ):
    '''
    d: volume density, shape: [x, y, z], type: float32, range: [0, 1]
    J: scalar field
    ori: fiber orientation, shape [x, y, z, 3]
    ed: filter threshold for density
    eJ: filter threshold for J
    '''
    bg = np.logical_or(d < ed, J < eJ)
    d[bg] = 0.0
    ori[bg] = 0.0


def denoiseFelt():
    d = np.load(r'D:\Document\clothSimulation\Dataset\felt\felt.dat')  # [105, 1013, 992], 'uint16', [0, 65535]
    d = d.astype(dtype='float32')
    d /= 65535.0
    print d.shape, d.dtype

    # load J and orientation
    j0 = r'D:\Document\clothSimulation\Dataset\felt\J0_100.dat'
    j1 = r'D:\Document\clothSimulation\Dataset\felt\J100_200.dat'
    j2 = r'D:\Document\clothSimulation\Dataset\felt\J200_300.dat'
    j3 = r'D:\Document\clothSimulation\Dataset\felt\J300_400.dat'
    j4 = r'D:\Document\clothSimulation\Dataset\felt\J400_500.dat'
    j5 = r'D:\Document\clothSimulation\Dataset\felt\J500_508.dat'
    
    j0 = Util.loadData(j0)   # [1013, 992, 100]
    j1 = Util.loadData(j1)
    j2 = Util.loadData(j2)
    j3 = Util.loadData(j3)
    j4 = Util.loadData(j4)
    j5 = Util.loadData(j5)   # [1013, 992, 8]

    J = np.concatenate((j0, j1, j2, j3, j4, j5), axis=2)
    J = J.swapaxes(0, 2).swapaxes(1, 2)
    print J.shape, J.dtype

    del j0, j1, j2, j3, j4, j5

    ed = 0.5
    eJ = 0.5
    bg = np.logical_or(d < ed, J < eJ)
    d[bg] = 0.0

    print 'saving denosing density ...'
    felt_dn_path = r'D:\Dataset\felt\felt_dn.dat'
    Util.dumpData(d, felt_dn_path)

#    del d, J
#
#    ori0 = r'D:\Document\clothSimulation\Dataset\felt\ori0_100.dat'
#    ori1 = r'D:\Document\clothSimulation\Dataset\felt\ori100_200.dat'
#    ori2 = r'D:\Document\clothSimulation\Dataset\felt\ori200_300.dat'
#    ori3 = r'D:\Document\clothSimulation\Dataset\felt\ori300_400.dat'
#    ori4 = r'D:\Document\clothSimulation\Dataset\felt\ori400_500.dat'
#    ori5 = r'D:\Document\clothSimulation\Dataset\felt\ori500_508.dat'
#    
#    ori0 = Util.loadData(ori0) # [1013, 992, 100, 3]
#    ori1 = Util.loadData(ori1)
#    ori2 = Util.loadData(ori2)
#    ori3 = Util.loadData(ori3)
#    ori4 = Util.loadData(ori4)
#    ori5 = Util.loadData(ori5)
#
#    ori = np.concatenate((ori0, ori1, ori2, ori3, ori4, ori5), axis=2)
#    ori = ori.swapaxes(0, 2).swapaxes(1, 2)
#    print ori.shape, ori.dtype
#
#    del ori0, ori1, ori2, ori3, ori4, ori5
#
#    ori[bg[0:100]] = 0.0
#    Util.dumpData(ori[0:100], r'D:\Dataset\felt\ori0_100_dn.dat') 
#    print '1'
#    ori[bg[100:200]] = 0.0
#    Util.dumpData(ori[100:200], r'D:\Dataset\felt\ori100_200_dn.dat')
#    print '2'
#    ori[bg[200:300]] = 0.0
#    Util.dumpData(ori[200:300], r'D:\Dataset\felt\ori200_300_dn.dat')
#    print '3'
#    ori[bg[300:400]] = 0.0
#    Util.dumpData(ori[300:400], r'D:\Dataset\felt\ori300_400_dn.dat')
#    ori[bg[400:500]] = 0.0
#    Util.dumpData(ori[400:500], r'D:\Dataset\felt\ori400_500_dn.dat')
#    ori[bg[500:508]] = 0.0
#    Util.dumpData(ori[500:508], r'D:\Dataset\felt\ori500_508_dn.dat')

    print 'finish denoising felt ...'


def denoiseSilk():
    d = np.load(r'')  # [508, 1013, 992], 'uint16', [0, 65535]
    J = np.load(r'') # [1013, 992, 105], 'float32'
    ori = np.load(r'D:\Dataset\silk\orientation.dat') # [1013, 992, 105, 3], 'float32'
    ed = 0.4
    eJ = -1.0

    # preprocess
    d = d.astype(dtype='float32')
    d /= 65535.0
#    print d.shape, d.dtype, d.min(), d.max()

    J = J.swapaxes(0, 2).swapaxes(1, 2)
#    print J.shape, J.dtype, J.min(), J.max()

    ori = ori.swapaxes(0, 2).swapaxes(1, 2)
#    print ori.shape, ori.dtype

    # denoising
    denoise(d, J, ori, ed, eJ)
    print 'finish deonising ...'

    # save
    silk_dn_path = r'D:\Dataset\silk\silk_dn.dat'
    ori_dn_path = r'D:\Dataset\silk\orientation_dn.dat'

    Util.dumpData(d, silk_dn_path)
    Util.dumpData(ori, ori_dn_path)


if __name__ == '__main__':
#    denoiseFelt()
    pass
#-----------------------------------------

# RecoverOrientation.py -------------------
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 04 10:10:25 2015

@author: yalong.li
"""

from scipy import ndimage
import numpy as np
import matplotlib.pylab as plt
import Util
from scipy import signal

def generateBoundingPoints(nx, ny, nz):
    x = np.arange(0, nx)
    y = np.arange(0, ny)
    z = np.arange(0, nz)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')      

    xv_l = (xv == 0)
    xv_r = (xv == nx-1)
    xv_bound = np.logical_or(xv_l, xv_r)
    
    yv_b = (yv == 0)
    yv_f = (yv == ny-1)
    yv_bound = np.logical_or(yv_b, yv_f)
    
    zv_b = (zv == 0)
    zv_u = (zv == nz-1)
    zv_bound = np.logical_or(zv_b, zv_u)
    
    bound = np.logical_or(xv_bound, yv_bound)
    bound = np.logical_or(bound, zv_bound)
    
    xv = xv[bound]
    yv = yv[bound]
    zv = zv[bound]
    
    return np.column_stack((xv.ravel(), yv.ravel(), zv.ravel()))


#def generatePoints(nx, ny, nz):
#    '''
#      y
#    00400
#   x1   3
#    02000
#    '''
#    xm = nx / 2
#    ym = ny / 2
#    zm = nz / 2
#    
#    x = np.arange(-xm, xm + 1)
#    x = x.reshape((x.size, 1))
#    y = np.ones_like(x) * (-ym)
#    p1 = np.column_stack((x, y))
#    
#    x = np.arange(-xm, xm + 1)
#    x = x.reshape((x.size, 1))
#    y = np.ones_like(x) * (ym)
#    p2 =  np.column_stack((x, y))
#    
#    y = np.arange(-ym + 1, ym)
#    y = y.reshape((y.size, 1))
#    x = np.ones_like(y) * (-xm)
#    p3 = np.column_stack((x, y))
#    
#    y = np.arange(-ym + 1, ym)
#    y = y.reshape((y.size, 1))
#    x = np.ones_like(y) * (xm)
#    p4 = np.column_stack((x, y))
#
#  
#    p_xy = np.concatenate((p1, p2, p3, p4))   
#    z = np.arange(-zm, zm + 1)
#
#    size_xy = p_xy.shape[0]
#    size_z = z.shape[0]    
#    
#    z = z.reshape((z.size, 1))
#    z = np.tile(z, (1, size_xy))
#    z = z.reshape((z.size, 1))
#    
#    p_xy = np.tile(p_xy, (size_z, 1))
#    
#    p = np.column_stack((p_xy, z))
#    
#    #print p
#    print p.shape[0]
#    return p


#def gaussian(x, u=0, s=1):
#    return (1/(np.sqrt(2*np.pi)*s))*np.exp(-0.5*((x-u)/s)**2)


def diffOfGaussian(s, t, r_sq):
#    return -2*gaussian(-s*r_sq) + gaussian(-t*r_sq)
    return -2 * np.exp(-s * r_sq) + np.exp(-t * r_sq)


def calculateFilterQ(q, h, s, t, d, c):
    for (xi, yi, zi) in np.ndindex(q.shape):
        p = np.asarray([xi, yi, zi], dtype='float64')
        cp = p - c
        r_sq = (cp**2).sum() - (np.dot(cp, d))**2
        if np.abs(r_sq) < 1.0e-10:
            r_sq = 0.0
            # when cp, d parallel, r_sq very small, but why negative??
        q[xi,yi,zi] = -2 * np.exp(-s * r_sq) + np.exp(-t * r_sq)
        

def testDiffOfGaussian(s=4, t=3):
    x = np.linspace(-5, 5, 100)
    y = np.zeros_like(x)
    
    for i in range(x.shape[0]):
        y[i] = diffOfGaussian(s, t, x[i]*x[i])
    
    
    plt.plot(x, y)
    return y


def precomputeFilterQ():
    # Precompute kernel 'q' on a set of directions

    # Calculate the directions
    nx = 32
    ny = 32
    nz = 6
    bound_points = generateBoundingPoints(nx + 1, ny + 1, nz + 1)
    center = np.array([nx/2, ny/2, nz/2])
    
    directions = bound_points - center
    directions = np.asarray(directions, dtype='float64')    

    # normalize    
    for i in range(0, directions.shape[0]):
        sqrt_sum = np.sqrt((directions[i]**2).sum())
        directions[i] = directions[i] / sqrt_sum
    
    d_path = r'D:\Datasets\RecoverOrientationData\ds_nx32ny32nz6.dat'
    Util.dumpData(directions, d_path)    
    
    # Calculate 'q', material: silk
    h = 12
    s = 4 #!! some erros in paper about these parameters, s>t!!
    t = 3
    c = np.array([h/2, h/2, h/2], dtype='float64') # center
    d = directions
    
    q = np.zeros((d.shape[0], h, h, h))

    for i in range(0, q.shape[0]):
        calculateFilterQ(q[i], h, s, t, d[i], c)    # pass reference?!!

    qs_path = r'D:\Datasets\RecoverOrientationData\qs_h12s4t3.dat'
    Util.dumpData(q, qs_path) 


def binarizeVolume():
    volume = Util.loadData(r'D:\Datasets\PreprocessingData\silk.dat')
    volume = (volume + 0.0) / 65535.0

    eplsion_d = 0.4
    
    fiber = volume >= eplsion_d
    background = np.logical_not(fiber)
    
    volume[fiber] = 0.0
    volume[background] = 1.0
    
    # reshape to [h, w, n]
    volume = volume.swapaxes(0, 2).swapaxes(0, 1)

    bi_p = r'D:\Datasets\RecoverOrientationData\volumeBin_04.dat'
    Util.dumpData(volume, bi_p)


def main():
    print 'loading data ...'
    ds = Util.loadData(r'E:\yalongli\Projects\RecoverOrientationData\ds_nx32ny32nz6.dat')
    qs = Util.loadData(r'E:\yalongli\Projects\RecoverOrientationData\qs_h12s4t3.dat')
    f = Util.loadData(r'E:\yalongli\Projects\RecoverOrientationData\volumeBin_04.dat')
    
    # small test volume
    f = f[:,:,75:85]
    print f.shape
    
    # Calculate 'J' on these 'q's one by one and store the max_J and max_q in
    # non-empty cells
    print 'initialzing J_max, d_max ...'
    inf = -999999.0
    J_max = np.ones_like(f) * inf
    d_max = np.zeros((f.shape[0], f.shape[1], f.shape[2], 3))
    
    print 'calculating J, d ...'
    for i in range(qs.shape[0]):
        J = ndimage.correlate(f, qs[i], mode='constant', cval=1.0) # background = 1.0!
        update = J > J_max
        J_max[update] = J[update]
        d_max[update] = ds[i]
        if i%100 == 0:
            print i, '...'
    
    # save non-empty cells' result, set empyt cells' direction to zero
    background = (f > 0.9)     
    d_max[background] = 0.0
    
    print 'saving ...'
    ori_p = r'E:\yalongli\Projects\RecoverOrientationData\orientationXYZ7585.dat'
    J_p = r'E:\yalongli\Projects\RecoverOrientationData\JXYZ7585.dat'
    Util.dumpData(d_max, ori_p)
    Util.dumpData(J_max, J_p)


def fftConv():
    print 'loading data ...'
    ds = Util.loadData(r'E:\yalongli\Projects\RecoverOrientationData\ds_nx32ny32nz6.dat')
    qs = Util.loadData(r'E:\yalongli\Projects\RecoverOrientationData\qs_h12s4t3.dat')
    f = Util.loadData(r'E:\yalongli\Projects\RecoverOrientationData\volumeBin_04.dat') # float32?

    # padding f with 1s
    h = 12 # for silk
    pad_len = (h + 2) / 2
    np.pad(f, (pad_len,), 'constant', constant_values(1.0, 1.0)) # 1.0 - background, 0.0 - fiber

    print 'initialzing J_max, d_max ...'
    inf = -999999.0
    J_max = np.ones_like(f, dtype='float32') * inf
    d_max = np.zeros((f.shape[0], f.shape[1], f.shape[2], 3), dtype='float32')
    
    print 'calculating J, d ...'
    for i in range(qs.shape[0]):
#        J = ndimage.correlate(f, qs[i], mode='constant', cval=1.0) # background = 1.0!
        update = J > J_max
        J_max[update] = J[update]
        d_max[update] = ds[i]
        if i%100 == 0:
            print i, '...'

def testConvolution():
    a = np.array([[1, 2, 0, 0],
                 [5, 3, 0, 4],
                 [0, 0, 0, 7],
                 [9, 3, 0, 0]], dtype='float32')
    
    k = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype='float32')
    
    r = signal.fftconvolve(a, k, mode='same')
    print r

def testCorrelate():
    a = np.array([[1, 2, 0, 0],
                 [5, 3, 0, 4],
                 [0, 0, 0, 7],
                 [9, 3, 0, 0]])
    
    k = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]])

    a = a[a > 0]    
    
    b = ndimage.correlate(a, k, mode='constant', cval=0.0)
    
    print b
    
    # 2
    a = np.arange(0, 27).reshape((3, 3, 3))
    k = np.array([1, 1, 1])


if __name__ == '__main__':
#    print 'binarizing ...'
#    binarizeVolume()
#    print 'precomputing filter qs ...'
#    precomputeFilterQ()
#    main()#17:50
    pass
#-----------------------------------------

# Main2.py---------------------
'''Image processing code of implementing the paper "Building Volumetric
Appearance Models of Fabric using Micro CT Imaging.
'''
import numpy as np
from TiffsHandler import TiffsHandler
import Util
import os


def polyFitVolume(volume):    
    nx, ny, nz = volume.shape
    x = np.arange(0, nx)
    y = np.arange(0, ny)
    xv, yv = np.meshgrid(x, y, indexing='ij')    
    
    xv = xv.ravel()
    yv = yv.ravel()
    
    zv = np.zeros_like(xv)   
    idx = np.arange(0, xv.size)
    
    w = np.zeros_like(xv, dtype='float64')
    
    for i, xi, yi in zip(idx, xv, yv):
        total_density = volume[xi,yi,:].sum()
        w[i] =  total_density
        p = 0.65
        total_density *= p
        
        for k in range(0, nz):
            cum_density = volume[xi,yi,nz-1-k:nz].sum()
            if cum_density >= total_density:
                zv[i] = nz - 1 - k
                break
   
    #w = np.sqrt(volume).ravel()
    in_circle = w > 28.7
    xv = xv[in_circle]
    yv = yv[in_circle]
    zv = zv[in_circle]
    
    m = Util.poly2ndfit2d(xv, yv, zv, w=None)
    
    return m


def plot2ndPolySurface(m):    
    nx, ny = (1013, 992)
    x = np.arange(0, nx) 
    y = np.arange(0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    Z = Util.poly2ndval2d(X, Y, m)
    Util.plotSurface(X, Y, Z)
    
    return Z


def resampleVolume(volume, m):
    nx, ny, nz = volume.shape
    x = np.arange(0, nx)
    y = np.arange(0, ny)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    
    Z = Util.poly2ndval2d(xv, yv, m)

    # transform to [0, Z.max - Z.min]
    Zmin = Z.min()
    Z = Z - Zmin
    
    Zmax = Z.max()

    R = np.zeros((nx, ny, nz + int(Zmax + 0.5)), dtype='float32')
    
    for i, j in zip(xv.ravel(), yv.ravel()):
        s = int(Zmax + 0.5) - int(Z[i, j] + 0.5)
        R[i, j, s:s+nz] = volume[i, j, :]
    
    return R


def straightenSample():
    volume = Util.loadData(r'E:\yalongli\Projects\PreprocessingData\silk.dat')
    volume = volume.swapaxes(0, 2).swapaxes(0, 1)
    
    volume = volume + 0.0
    volume /= 65535.0
    
    print volume.shape
    
    print 'fitting ...'
    m = polyFitVolume(volume)

#    print 'resampling ...'
#    R = resampleVolume(volume, m)
    
    print m
#    Util.dumpData(R, r'D:\Datasets\PreprocessingData\silk_rs29.dat')


def saveTiffs():
    R = Util.loadData(r'D:\Datasets\PreprocessingData\silk_rs29.dat')
    R = (R * 65535).astype('uint16')
    R = R.swapaxes(0, 2).swapaxes(1, 2)
    f_p = r'D:\Datasets\PreprocessingData\silkrs29'
    TiffsHandler.writeTiffImagesInFolder(R, f_p, prefix='silk')


def processing():
    mat_name = r'silk'
    ori_folder = r'D:\Datasets\silkA_16bit\16bit'
    res_folder = r'D:\Datasets\PreprocessingData'

    # Read tiff images and dump them into a binary file
#    print 'Read and Dump images begins ...'    
#    volume_path = os.path.join(res_folder, mat_name + '.dat')
#    volume_data = TiffsHandler.readTiffImagesInFolder(ori_folder)  
#    print 'Read images finished ...'
#    
#    print 'Dump volume to (%s) begins ...' % volume_path
#    Util.dumpData(volume_data, volume_path)
#    print 'Dump volume to (%s) finished ...' % volume_path

    # load volume data
    volume_path = os.path.join(res_folder, mat_name + '.dat')
    volume_data = Util.loadData(volume_path)
    
    # Histogram equalize
    print 'Histogram equalize begins ...'
    bins = np.arange(0, 65537)
#    bins = np.linspace(0, 65537, 50, endpoint=True)
    volume_he = Util.histEq(volume_data, bins=bins)
    volume_he = volume_he.astype('float32')
    print 'Histogram equalize finished ...'
    
    volume_he_path = os.path.join(res_folder, mat_name + '_he.dat')
    print 'Dump volume_he to (%s) begins ...' % volume_he_path
    Util.dumpData(volume_he, volume_he_path)
    print 'Dump volume_he to (%s) finished ...' % volume_he_path    
    
    return volume_data, volume_he
    

if __name__ == '__main__':
    pass
    #straightenSample()
#--------------------------------------------


#---- fitvolume oooold ------
def fitVolume():
    volume = Util.loadData(r'D:\Datasets\PreprocessingData\silk_he.dat')
    volume = volume.swapaxes(0, 2).swapaxes(0, 1)
    
    t = volume.shape[2]
    idx = np.arange(0,t,2)
    volume = volume[:,:,idx]
    
    nx, ny, nz = volume.shape
    x = np.arange(0, nx)
    y = np.arange(0, ny)
    z = np.arange(0, nz)
#    xv, yv, zv = np.meshgrid(x, y, z)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')    
    
#    xv = xv.ravel()
#    yv = yv.ravel()
#    zv = zv.ravel()
    del x, y, z
    
#    m = Util.polyfit2d(xv, yv, zv, w=volume[xv, yv, zv])
    m = Util.polyfit2d(xv.ravel(), yv.ravel(), zv.ravel(), w=volume.ravel())
    
    return m


def testFitVolume():
#    volume = Util.loadData(r'D:\Datasets\PreprocessingData\silk_he.dat')
#    volume = volume.swapaxes(0, 2).swapaxes(0, 1)
    m = [2.30025646e+01, 1.20812555e-02, -1.10236229e-05, 3.25151389e-03,
  -7.60125483e-07, 1.00304238e-08, -2.44075796e-06, -1.05850001e-08,
   2.62056002e-13]
    m = np.asarray(m)    
    
    nx, ny = (1013,992)
    x = np.arange(0, nx)
    y = np.arange(0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = Util.polyval2d(X, Y, m)

    print Z[0, 0], Z[0, 1]
    Z[0, 0] = 0
    Z[0, 1] = 105

    Util.plotSurface(X, Y, Z)
    return Z


def resample():
    volume = Util.loadData(r'D:\Datasets\PreprocessingData\silk_he.dat')
    volume = volume.swapaxes(0, 2).swapaxes(0, 1)

    m = [25.6207523,
         0.00563881328,
         -7.50831215e-06,
         -0.000110252038,
         5.8775209e-07,
         -4.20923367e-10,
         2.7086295e-07,
         -1.18467307e-09,
         1.03793103e-12]
    m = np.asarray(m)
    
    nx, ny, nz = volume.shape
    x = np.arange(0, nx)
    y = np.arange(0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = Util.polyval2d(X, Y, m)
    
    print Z.shape
    
    Zmin = Z.min()
    Zmax = Z.max()
    
    R = np.zeros((nx, ny, nz + Zmax))
    
    for i, j in zip(X.ravel(), Y.ravel()):
        s = int(Zmax - Z[i, j])
        R[i, j, s:s+nz] = volume[i, j, :]
    
    Util.dumpData(R, r'D:\Datasets\PreprocessingData\silk_rs.dat')
        
    return R
    
